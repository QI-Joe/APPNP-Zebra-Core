import argparse
import queue
from typing import Optional

from torch_geometric.loader import NeighborLoader
import torch.nn.functional as F
from torch import NoneType, Tensor
import math
import torch
from utils.Sparse_operation import PyG_to_Adj
from torch_geometric.data import Data
from torch_sparse import spmm as spmm2
# from utils import create_propagator_matrix

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, SparseTensor
from torch_geometric.utils import is_torch_sparse_tensor, spmm, to_edge_index
from torch_geometric.utils.sparse import set_sparse_value
from typing import Union
import numpy as np
# from utils.my_NeighborLoader import Fixed_NeighborLoader
from utils.uselessCode import create_propagator_matrix, TPPR_Simple, Running_Permit, node_index_anchoring
from utils.my_dataloader import Temporal_Dataloader
import time
import os

def APPNP_config(model_details):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="cora", help='Dataset to use.')
    parser.add_argument('--snapshot', type=int, default=4, help='Snapshot to use.')
    parser.add_argument('--epoch_train', type=int, default=1500, help='Number of epochs to train.')
    parser.add_argument('--alpha', type=float, default=0.1, help='Teleport probability.')
    parser.add_argument('--hidden', type=int, default=256, help='Number of hidden units.')
    parser.add_argument("--K", type=int, default=15, help="Number of iterations.")
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--neighbor_sample_size', type=int, default=400, help='Number of largest round neighbors to sample.')

    argrs = parser.parse_args(model_details)
    return argrs

class Memory_dense(torch.nn.Module):
    def __init__(self, ori_weight, old_bias, ori_shape: tuple[int, int], *args, **kwargs) -> None:
        super(Memory_dense, self).__init__(*args, **kwargs)
        self.past_weight = ori_weight
        self.past_bias = old_bias
        self.past_shape = ori_shape

    def matrix_re_init(self, new_weight, new_bias):
        with torch.no_grad():
            new_weight[:self.past_shape[0], :self.past_shape[1]] = self.past_weight
            new_bias[:self.past_shape[1]] = self.past_bias
        return new_weight, new_bias

def uniform(size, tensor):
    """
    Uniform weight initialization.
    :param size: Size of the tensor.
    :param tensor: Tensor initialized.
    """
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)

class DenseFullyConnected(torch.nn.Module):
    """
    Abstract class for PageRank and Approximate PageRank networks.
    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param density: Feature matrix structure.
    """
    def __init__(self, in_channels, out_channels):
        super(DenseFullyConnected, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining the weight matrices.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(self.out_channels))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        uniform(self.out_channels, self.bias)

    def forward(self, features):
        """
        Doing a forward pass.
        :param features: Feature matrix.
        :return filtered_features: Convolved features.
        """
        filtered_features = torch.mm(features, self.weight_matrix)
        filtered_features = filtered_features + self.bias
        return filtered_features

class SparseFullyConnected(torch.nn.Module):
    """
    Abstract class for PageRank and Approximate PageRank networks.
    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param density: Feature matrix structure.
    """
    def __init__(self, in_channels, out_channels):
        super(SparseFullyConnected, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining the weight matrices.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(self.out_channels))

    def kept_past(self):
        weight = self.weight_matrix.detach().clone()
        bias = self.bias.detach().clone()
        return Memory_dense(weight, bias, self.weight_matrix.shape)

    def load_past(self, memory: Memory_dense):
        self.weight_matrix, self.bias = memory.matrix_re_init(self.weight_matrix, self.bias)

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        uniform(self.out_channels, self.bias)

    def forward(self, features):
        filtered_features = torch.mm(features, self.weight_matrix)
        return filtered_features + self.bias

    def forward2(self, feature_indices, feature_values):
        """
        Making a forward pass.
        :param feature_indices: Non zero value indices.
        :param feature_values: Matrix values.
        :return filtered_features: Output features.
        """
        number_of_nodes = torch.max(feature_indices[0]).item()+1
        number_of_features = torch.max(feature_indices[1]).item()+1
        filtered_features = spmm2(index = feature_indices,
                                 value = feature_values,
                                 m = number_of_nodes,
                                 n = number_of_features,
                                 matrix = self.weight_matrix)
        filtered_features = filtered_features + self.bias
        return filtered_features

class APPNP(MessagePassing):
    r"""The approximate personalized propagation of neural predictions layer
    from the `"Predict then Propagate: Graph Neural Networks meet Personalized
    PageRank" <https://arxiv.org/abs/1810.05997>`_ paper.

    .. math::
        \mathbf{X}^{(0)} &= \mathbf{X}

        \mathbf{X}^{(k)} &= (1 - \alpha) \mathbf{\hat{D}}^{-1/2}
        \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \mathbf{X}^{(k-1)} + \alpha
        \mathbf{X}^{(0)}

        \mathbf{X}^{\prime} &= \mathbf{X}^{(K)},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Args:
        K (int): Number of iterations :math:`K`.
        alpha (float): Teleport probability :math:`\alpha`.
        dropout (float, optional): Dropout probability of edges during
            training. (default: :obj:`0`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F)`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F)`
    """
    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, K: int, alpha: float, layer2: list, device, graph: Data, layer1: tuple = (0,0), dropout: float = 0.,
                 cached: bool = False, add_self_loops: bool = True, hook: bool = False,
                 hook_queue=None, mode: str= "exact", normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__()
        self.K = K
        self.alpha = alpha
        self.dropout = dropout
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.device = device
        self.hook=hook
        self.matrix_queue: Union[queue.Queue | NoneType] = hook_queue
        self.graph = graph
        self.mode = self.ppr_mode_distinct(mode=mode)
        self.T_shape = None

        if self.hook:
            self.customized_hook()

        if layer1[0] and layer1[0]:
            self.layer1 = SparseFullyConnected(*layer1).to(self.device)
        self.layer2 = DenseFullyConnected(*layer2).to(self.device)

        self._cached_edge_index = None
        self._cached_adj_t = None

        alpha_list, node_num, beta_list = kwargs["alphalist"], kwargs["node_num"], kwargs["betalist"]
        self.tppr_ = TPPR_Simple(alpha_list=alpha_list, node_num=node_num, beta_list=beta_list, topk=self.K)
        self.initial_propagator()
        self.updated: bool = False

    def ppr_mode_distinct(self, mode: str):
        mode = mode.split("-")
        if len(mode) == 0:
            raise KeyError(f"input order is wrong, the given input is {mode}")
        return mode

    def propagator_switch(self):
        self.propagator = self.last_propagator
        print(self.propagator.shape, self.new_propagator.shape)

    def tppr_idx_preapre(self, graph: Temporal_Dataloader):
        src_edges, dest_edges = graph.edge_index[0], graph.edge_index[1]
        # timestamps, edge_idxs = graph.edge_attr.cpu().numpy(), np.array(range(0, graph.edge_index.shape[1]))
        timestamps, edge_idxs = np.array(range(0, graph.edge_index.shape[1])), np.array(range(0, graph.edge_index.shape[1]))
        tppr_batch = np.concatenate([src_edges, dest_edges, dest_edges])
        return tppr_batch, timestamps, edge_idxs

    def tppr_entire_graph_updating(self, graph: Temporal_Dataloader):
        tppr_batch, timestamps, edge_idxs = self.tppr_idx_preapre(graph)
        permit = Running_Permit(event_or_snapshot="snapshot", ppr_updated=False)
        _, _, _, _ = self.tppr_.streaming_topk(source_nodes=tppr_batch, timestamps=timestamps, edge_idxs=edge_idxs, updated=permit)
        self.updated = True

    def tppr_updating(self, training: bool, graph):
        r"""
        updating the propagator

        Condition:
        - if in training model and last_propagator is None, means the model is in the first epoch and propagaor 
        is not updated, so pass if and go updating
        - if in testing model and new_propagator is None, means the model is in epoch and propagaor
        for T+1 moment is not updated, so pass if and go updating
        - if in training model and last_propagator is not None, means the model is in the middle of training, so
        reverse the self.propagator to T moment 
        - if in testing model and new_propagator is not None, means the model is in the middle of testing, so
        switch the self.propagator to T+1 moment
        """
        if self.last_propagator!=None and training:
            self.propagator = self.last_propagator
            return
        elif self.new_propagator!=None and not training:
            self.propagator = self.new_propagator
            return

        if len(self.mode) > 1: 
            self.setup_tppr_propagator(graph, self.mode, training)
        if self.mode[0] == "exact":
            raise KeyError("Since graph is expected to in Temproral Graph type, exact is not avalibale anymore")

    def end_temporal(self):
        self.last_propagator = self.new_propagator
        self.new_propagator = None

    def reset_parameters(self):
        super().reset_parameters()
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        self._cached_edge_index = None
        self._cached_adj_t = None

    def customized_hook(self):
        from utils.Test_matrix import get_queue
        self.barrier, self.lock = get_queue()
        print("TGNN modified code matrix address", id(self.matrix_queue))

    def initial_propagator(self):
        r"""
        the difference of initial_propagator and setup_propagator is
        initial_propagator is used to initialize the memory cache of propagator matrix,
        :param new_propagator is used for T+1 moment of propagator matrix
        :param last_propagator is used for T moment propagator matrix
        :param when the model is finished for a temporal, last_propagator will be updated to new_propagator, new propagator will be set as None
        """
        self.last_propagator: Union[None|Tensor] = None
        self.new_propagator: Union[None|Tensor] = None

    def setup_propagator(self):
        """
        Creating propagation matrix (Personalized Pagrerank or adjacency).
        """
        self.propagator = create_propagator_matrix(self.graph.edge_index, self.alpha)
        if self.mode[0] == "exact":
            self.propagator = self.propagator.to(self.device)
        else:
            self.edge_indices = self.propagator["indices"].to(self.device)
            self.edge_weights = self.propagator["values"].to(self.device)

    def setup_tppr_propagator(self, graph: Temporal_Dataloader, mode: list[str], training: bool):
        """
        purely huge and useless code, nothing to do but to make code more like a huge garbage mountain
        """
        if mode[1] == "fora":
            self.propagator = torch.load(r"logs/temporary_running_result/fora_cora.pt").to(self.device)
        elif mode[1] == "appro":
            self.propagator = torch.load(r"logs/temporary_running_result/Matrix_approx_cora.pt").to(self.device)
        else: 
            if training:
                # this condition will only be used in the first epoch of training
                self.T_shape = graph.edge_index.shape[1]
                kwargs = {"full_update": True, "tranucate": 0, "updated": self.updated}
            else:
                kwargs = {"full_update": False, "tranucate": self.T_shape, "updated": self.updated}
                self.T_shape = graph.edge_index.shape[1]
            kwargs["nodes"] = self.graph.num_nodes
            self.propagator = TPPR_invoking(tppr=self.tppr_, graph_data=graph, kwargs= kwargs).to_dense().to(self.device)
        if training:
            self.last_propagator = self.propagator
        else:
            self.new_propagator = self.propagator

    def init_layer1(self, layer1):
        self.layer1 = SparseFullyConnected(*layer1).to(self.device)

    def reload_layer1(self, new_layer1: tuple[int, int], memory: Memory_dense):
        self.layer1 = SparseFullyConnected(*new_layer1).to(self.device)
        self.layer1.load_past(memory)

    def forward(self, feature_values, edge_index=None):
        r"""
        Making a forward propagation pass.
        :param feature_indices: Feature indices for feature matrix.
        :param feature_values: Values of the normalized adjacency matrix.
        :return self.predictions: Predicted class label log softmaxes.
        """
        latent_features_1 = self.layer1(feature_values)

        latent_features_1 = torch.nn.functional.relu(latent_features_1)

        latent_features_1 = torch.nn.functional.dropout(latent_features_1,
                                                        p=self.dropout,
                                                        training=self.training)

        latent_features_2: torch.Tensor = self.layer2(latent_features_1)

        if self.hook:
            self.lock.acquire()
            self.matrix_queue.put(("tgnn_src", feature_values.clone().cpu()))
            self.matrix_queue.put(("tgnn_layer2", latent_features_2.detach().clone().cpu()))
            if not isinstance(self.propagator, torch.Tensor):
                raise TypeError("propagator not in expected Tensor format, got ", type(self.propagator))
            self.matrix_queue.put(("tgnn_propagator", self.propagator.detach().clone().cpu()))
            self.lock.release()

        if self.mode[0] == "exact" or self.mode[0] == "tppr":
            self.predictions = torch.nn.functional.dropout(self.propagator,
                                                           p=self.dropout,
                                                           training=self.training)

            self.predictions = torch.mm(self.predictions, latent_features_2)

        # I dont think this part could be used in the future
        elif self.mode[0] == "torch_tppr":
            
            x = latent_features_2
            edge_weight = None
            if self.normalize:
                if isinstance(edge_index, Tensor):
                    cache = self._cached_edge_index
                    if cache is None:
                        edge_index, edge_weight = gcn_norm(  # yapf: disable
                            edge_index, edge_weight, x.size(self.node_dim), False,
                            self.add_self_loops, self.flow, dtype=x.dtype)
                        if self.cached:
                            self._cached_edge_index = (edge_index, edge_weight)
                    else:
                        edge_index, edge_weight = cache[0], cache[1]

                elif isinstance(edge_index, SparseTensor):
                    cache = self._cached_adj_t
                    if cache is None:
                        edge_index = gcn_norm(  # yapf: disable
                            edge_index, edge_weight, x.size(self.node_dim), False,
                            self.add_self_loops, self.flow, dtype=x.dtype)
                        if self.cached:
                            self._cached_adj_t = edge_index
                    else:
                        edge_index = cache

            h = x
            for k in range(self.K):
                if self.dropout > 0 and self.training:
                    if isinstance(edge_index, Tensor):
                        if is_torch_sparse_tensor(edge_index):
                            _, edge_weight = to_edge_index(edge_index)
                            edge_weight = F.dropout(edge_weight, p=self.dropout)
                            edge_index = set_sparse_value(edge_index, edge_weight)
                        else:
                            assert edge_weight is not None
                            # edge_weight = F.dropout(edge_weight, p=self.dropout)
                    else:
                        value = edge_index.storage.value()
                        assert value is not None
                        value = F.dropout(value, p=self.dropout)
                        edge_index = edge_index.set_value(value, layout='coo')

                # propagate_type: (x: Tensor, edge_weight: OptTensor)
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
                x = x * (1 - self.alpha)
                x = x + self.alpha * h
            self.predictions = x

        
        if self.hook:
            self.lock.acquire()
            self.matrix_queue.put(("tgnn_final", self.predictions.detach().clone().cpu()))
            self.lock.release()
            # self.barrier.wait()

        self.predictions = torch.nn.functional.log_softmax(self.predictions, dim=1)
        return self.predictions

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(K={self.K}, alpha={self.alpha})'

def TPPR_invoking(tppr: TPPR_Simple, graph_data: Temporal_Dataloader, kwargs: dict):
    all_node_size = kwargs["nodes"]
    del kwargs["nodes"]
    tppr_node, tppr_weight = tppr_matrix_computing(full_data=graph_data, tppr = tppr, **kwargs)
    flatten_tppr_list = graph_data.test_fast_sparse_build(tppr_node, tppr_weight)
    # at here, the sparse matrix should not be temporal graph size, but ALL GRAPH SIZE
    _, tppr_sparse = tppr2matrix(flatten_tppr_list, N=all_node_size)
    
    anchor_node_num = tppr_node.shape[0]
    anchor_ppr_save(tppr_sparse.to_dense(), anchor_nodes=anchor_node_num, edge_len=graph_data.edge_index.shape[1])

    if tppr_sparse.device == torch.device("cuda"):
        tppr_sparse = tppr_sparse.to("cpu")
    return tppr_sparse

def tppr2matrix(flatten_indices, N: int)->Union[torch.Tensor|np.ndarray]:
    indices_and_value = torch.Tensor(flatten_indices).T
    indices = indices_and_value[:2, :].to(torch.int64)
    simulate_ppr_value = indices_and_value[2, :].to(torch.float32)
    values = torch.ones((indices.shape[1], ), dtype=torch.float32)
    tppr_adj = torch.sparse_coo_tensor(indices, values, torch.Size([N, N]))
    tppr_adj_with_weight_sparse = torch.sparse_coo_tensor(indices, simulate_ppr_value, torch.Size([N, N]))

    return tppr_adj, tppr_adj_with_weight_sparse

def tppr_matrix_computing(full_data: Union[Data|Temporal_Dataloader], tppr: TPPR_Simple, full_update: bool, tranucate: int, updated: bool):
    permit = Running_Permit(event_or_snapshot="snapshot", ppr_updated=False)

    all_train_source, all_train_dest = full_data.ori_edge_index[0], full_data.ori_edge_index[1]
    # all_train_time, all_train_edgeIdx = full_data.edge_attr.cpu().numpy(), np.array(range(0, full_data.ori_edge_index.shape[1]))
    all_train_time, all_train_edgeIdx = np.array(range(0, full_data.ori_edge_index.shape[1])), np.array(range(0, full_data.ori_edge_index.shape[1]))
    input_source = np.concatenate([all_train_source, all_train_dest])

    if full_update:
        update_source = input_source
        timestamps, edge_idxs = all_train_time, all_train_edgeIdx
    else:
        tranucate_train_source, tranucate_train_dest = full_data.ori_edge_index[0, tranucate:], full_data.ori_edge_index[1, tranucate:]
        # tranucate_train_time, tranucate_train_edgeIdx = full_data.edge_attr.cpu().numpy()[tranucate:], np.array(range(tranucate, full_data.ori_edge_index.shape[1]))
        tranucate_train_time, tranucate_train_edgeIdx = np.array(range(tranucate, full_data.ori_edge_index.shape[1])), np.array(range(tranucate, full_data.ori_edge_index.shape[1]))
        calling_source = np.concatenate([tranucate_train_source, tranucate_train_dest])
        update_source = calling_source
        timestamps, edge_idxs = tranucate_train_time, tranucate_train_edgeIdx
    

    permit.set_all_true()
    
    tppr.streaming_topk(source_nodes=update_source, timestamps=timestamps, edge_idxs=edge_idxs, updated=permit)
    selected_node, selected_weight = tppr.single_extraction(input_source, timestamps=all_train_time)

    node_mask = node_index_anchoring(input_source)
    anchor_node, anchor_weight = selected_node[0][node_mask], selected_weight[0][node_mask]

    if anchor_node.shape[0] != anchor_weight.shape[0] != node_mask.shape[0]:
        raise ValueError("Anchor node length not right.")

    return anchor_node, anchor_weight


def data_split(data: Temporal_Dataloader):
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[:int(data.num_nodes*0.8)] = True
    data.train_mask = mask

    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[int(data.num_nodes*0.8):] = True
    data.val_mask = mask
    return data

def anchor_ppr_save(tppr: torch.Tensor, anchor_nodes: int, edge_len):
    """
    to_dense matrix
    """
    start_time = round(time.time(), 3)
    file_path = r"logs/tppr_anchor_matrix"
    file_name = rf"{anchor_nodes}_{edge_len}.pt"
    torch.save(tppr, os.path.join(file_path, file_name))


"""
else:
    localized_predictions = latent_features_2
    edge_weights = torch.nn.functional.dropout(self.edge_weights,
                                                p=self.dropout,
                                                training=self.training)

    for iteration in range(self.K):

        new_features = spmm2(index=self.edge_indices,
                            value=edge_weights,
                            n=localized_predictions.shape[0],
                            m=localized_predictions.shape[0],
                            matrix=localized_predictions)

        localized_predictions = (1-self.alpha)*new_features
        localized_predictions = localized_predictions + self.alpha*latent_features_2
    self.predictions = localized_predictions

    def loss(self, temporal_graph: Data, batch_size: int, neighbor_size, \
             loss_fn, optimizer, collect: bool = False, edge_index=None):
        neighborloader = Fixed_NeighborLoader(data=temporal_graph, big_fixed_batch_batch=2000, batch_size=batch_size, \
                                              num_neighbors=[-1], shuffle=False)
        entire_node = torch.arange(0, temporal_graph.x.size(0))
        train_indic = entire_node[temporal_graph.train_mask]

        if not collect:
            total_loss = 0
        else:
            emb = torch.FloatTensor([]).to(self.device)
        for idx, batch in enumerate(neighborloader):
            sub_batch_size = batch.batch_size
            sub_batch = batch.n_id
            train_seed_indic = torch.where(torch.isin(sub_batch, train_indic))[0].clone()

            normalized_adj_matrix = PyG_to_Adj(batch.edge_index.cpu(), neighbor_size).to(device=self.device)
            
            output = self.forward(feature_indices=batch.edge_index, feature_values=normalized_adj_matrix, edge_index=batch.edge_index.to("cuda:0"))
            # output shape : (batch_size, num_classes)
            # test for effect of incorrect normalized adj matrix to model

            if collect:
                emb = torch.cat((emb, output[:sub_batch_size]), dim=0)
            else:
                loss = loss_fn(output[train_seed_indic], temporal_graph.y[train_seed_indic])
                loss = loss+(0.005/2)*(torch.sum(self.layer2.weight_matrix**2))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        return total_loss/(idx+1) if not collect else emb

"""