from models.APPN import APPNP, APPNP_config, TPPR_invoking, data_split
import numpy as np
from utils.my_dataloader import Temporal_Dataloader, data_load, Temporal_Splitting, Dynamic_Dataloader, to_cuda
from Evaluation.time_evaluation import TimeRecord
import torch
from Evaluation.evaluate_nodeclassification import LogRegression, eval_model_Dy, eval_APPNP_st
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit
import queue
from utils.Sparse_operation import PyG_Cora_to_Sparse, PyG_to_Sparse, PyG_to_Adj
import argparse
import random
import copy

def main_APPN(args, time_: TimeRecord, hook_queue: queue = None, hook: bool = False):
    """
    fine-tune the TPPR code until accuracy converge to 85% or higher
    start from alpha_list and beta_list...    
    """
    time_.get_dataset(args.dataset)
    time_.set_up_logger(name="time_logger")
    time_.set_up_logger()
    time_.record_start()

    dataset = args.dataset
    snapshot = args.snapshot
    epoch_train = args.epoch_train
    alpha = args.alpha
    K = args.K
    hidden = args.hidden
    neighbor_tolerance = args.neighbor_sample_size
    alpha_list, beta_list = [0.4, 0.4], [0.5, 0.5]

    graph, _ = data_load(dataset)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    num_classes = graph.y.max().item()+1
    non_split, reset_idx = True, True

    temporal_list = Temporal_Splitting(graph).temporal_splitting(time_mode="view", reset_idx = reset_idx, \
                    snapshot=snapshot, views=snapshot-2, strategy="sequential", non_split=non_split)
    dataloader = Dynamic_Dataloader(temporal_list, graph)


    # At here, a matrix in shape of [all_node x hidden] will be initialized
    all_num_nodes: int = graph.num_nodes
    feature_in = graph.pos.shape[1]
    layer1 = (feature_in, hidden)
    layer2 = (hidden, num_classes)
    ppr_mode = "tppr-tppr"

    new_ppr = APPNP(K=K, alpha=alpha, layer1=layer1, layer2=layer2, dropout=0.5, device=device, \
                    graph=graph, mode=ppr_mode, hook=hook, hook_queue=hook_queue, \
                    alphalist=alpha_list, betalist=beta_list, node_num = all_num_nodes)
    prj = LogRegression(layer2[-1], num_classes).to(device)

    # new_ppr.tppr_entire_graph_updating(graph=graph)
    for t in range(3):
        time_.temporal_record()
        # dataloading has problem. Priority of debugging uncertain data loading method should put to first.
        
        temporal = dataloader.get_temporal()
        temporal_size = temporal.x.shape[0]

        all_num_nodes = temporal.num_nodes

        # updating TPPR on given new Temporal Graph 
        # Attention!!! TPPR requires clean if temporal graph is not splitted
        new_ppr.tppr_updating(training=True, graph=temporal)

        optimizer = torch.optim.Adam(new_ppr.parameters(), lr = 1e-2)

        random.seed(2024)
        torch.manual_seed(2024)
        transform = RandomNodeSplit(num_val=0.20,num_test=0.0)
        temporal: Temporal_Dataloader = transform(temporal)

        temporal = to_cuda(temporal)

        decoder, dense_feature = None, None
        bench_collector = []
        edge_idx = temporal.edge_index.to(device)
        dense_feature = temporal.x
        temporal_nodes = torch.tensor(temporal.my_n_id.node.node.values).to(device)
        print(temporal.num_nodes, temporal.num_edges, temporal_nodes.size()[0])

        for epoch in range(epoch_train):
            time_.epoch_record()
            new_ppr.train()
            prj.train()
            optimizer.zero_grad()
            
            node_embedding = new_ppr.forward(feature_values=dense_feature, edge_index=edge_idx)
            
            node_pred = node_embedding[temporal_nodes]
            loss = torch.nn.functional.cross_entropy(node_pred[temporal.train_mask], temporal.y[temporal.train_mask])
            loss = loss+(0.005/2)*(torch.sum(new_ppr.layer2.weight_matrix**2))

            loss.backward()
            optimizer.step()
            
            time_.epoch_end(temporal_size)
            if (epoch+1) % 25==0:
                new_ppr.eval()
                emb = new_ppr.forward(feature_values=dense_feature, edge_index=edge_idx)
                emb = emb[temporal_nodes]
                a, pred = emb[temporal.train_mask].max(dim=1)
                correct = pred.eq(temporal.y[temporal.train_mask]).sum().item()
                acc = correct / pred.size(0)
                print(f"APPNP Model | Epoch {epoch+1}, | loss {loss.item():4f}, train _aac {acc:4f}")
            if (epoch+1) % 200 == 0:
                eval_metrics, decoder = eval_APPNP_st(emb, temporal, num_classes, decoder, is_val=True, is_test=False)
                print(f"### -------------------------------------------------------------- \n \
                      Train Acc is: {acc:04f}, Val Acc is: {eval_metrics['accuracy']:04f}, train shape {pred.shape[0]}  \
                        \n### --------------------------------------------------------------")
                
                t1_temporal = copy.deepcopy(dataloader.get_T1graph(t))

                new_ppr.tppr_updating(training=False, graph=t1_temporal)

                t1_temporal = to_cuda(t1_temporal)
                t1_dense_feature = t1_temporal.x.to(device)
                t1_emb = new_ppr.forward(feature_values=t1_dense_feature, edge_index=edge_idx)
                
                test_metrics, _ = eval_model_Dy(t1_emb, t1_temporal, num_classes)
                # test_metrics = dict()
                test_metrics["train_acc"], test_metrics["val_acc"] = acc, eval_metrics["accuracy"]
                bench_collector.append(test_metrics)
                new_ppr.propagator_switch()

        time_.temporal_end(temporal.num_nodes)
        time_.score_record(bench_collector, temporal.num_nodes, t)
        print(f"here is the view {t+1}, with node number {temporal.num_nodes}")
        print(f"val acc {eval_metrics['test_acc']:03f}, \nval precision {eval_metrics['precision']:03f}, \nval recall {eval_metrics['recall']:03f}, \n")
        print(f"test acc {test_metrics['test_acc']:03f}, \ntest precision {test_metrics['precision']:03f}, \nval recall {test_metrics['recall']:03f}, \n")
        dataloader.update_event(t)
        new_ppr.end_temporal()
    
    time_.record_end()
    time_.to_log()
    print("PPR used is:", ppr_mode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='appnp')
    parser.add_argument('--device', type=str, default='cuda:0')
    default_param = {

    }

    # add hyper-parameters into parser
    param_keys = default_param.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}', type=type(default_param[key]), nargs='?')
    args, model_detail = parser.parse_known_args()
    print("Expected model is:", args.model.upper())
    args.model = args.model.lower()
    time_rec = TimeRecord(args.model)

    appnp_config = APPNP_config(model_detail)
    main_APPN(args=appnp_config, time_=time_rec)