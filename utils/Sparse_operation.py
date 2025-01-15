import torch_geometric as pyg
from torch_geometric.utils import to_dense_adj
import torch
import torch_sparse as sparse

def PyG_to_Sparse(edge_index: torch.Tensor, ds: str = "mathoverflow"):
    dense_matrix = to_dense_adj(edge_index)
    dense_matrix = torch.where(dense_matrix > 0, torch.tensor(1), dense_matrix).cpu()
    if len(dense_matrix.shape) > 2:
        dense_matrix = dense_matrix.squeeze()
    dense_len = dense_matrix.shape[0]
    for row in range(dense_len):
        denominator = torch.sum(dense_matrix[row, :])
        if denominator.item() == 0: continue
        dense_matrix[row] = dense_matrix[row] / denominator
    indices = torch.nonzero(dense_matrix)
    values = dense_matrix[indices[:, 0], indices[:, 1]]
    size = dense_matrix.shape
    return torch.sparse_coo_tensor(indices.t(), values, size)

def normalize_adj_torch(adj: torch.Tensor, I: torch.Tensor):
    """expected a symmetric adjacency matrix"""
    adj = adj + I
    degrees = adj.sum(axis=0)
    D = torch.diag(degrees, 0)
    D = D.to_sparse_coo().pow(-0.5)
    norm_adj = D @ adj @ D
    return norm_adj

def PyG_to_Adj(edge_idx: torch.Tensor, N: int):
    """
    go to check M = D^-1 * A * D^-1 for nomralization
    only data for 1 is not enough
    """
    if N is None:
        N = edge_idx.max().item() + 1
    assert edge_idx.shape[0] == 2, "edge_idx must have shape (2, E), but got {}".format(edge_idx.shape)
    sparse_shape = (N, N)
    element_values = torch.ones((edge_idx.shape[1], ))
    adj = torch.sparse_coo_tensor(edge_idx, values=element_values, size = sparse_shape).to_dense()
    adj = adj + adj.t().multiply(adj.t()>adj) - adj.multiply(adj.t()>adj)
    adj = normalize_adj_torch(adj , torch.eye(N))
    return adj.to_sparse_coo()

def PyG_Cora_to_Sparse(feature: torch.Tensor):
    padded_matrix = torch.zeros(2708, 2708)
    padded_matrix[:, :1433] = feature

    # Convert the padded matrix to a sparse matrix
    indices = torch.nonzero(padded_matrix)
    values = padded_matrix[indices[:, 0], indices[:, 1]]
    size = padded_matrix.shape
    return torch.sparse_coo_tensor(indices=indices.t(), values=values, size=size)

def manual_sparse(n_nodes: int, feature: list[list[int]]):
    """
    at here, we require feature is a 2-layer nested list, where feature[i] record connected node of i-th node
    """
    indices_row = [node for node in range(n_nodes) for _ in range(len(feature[node]))]
    indices_col = [fet for node in range(n_nodes) for fet in feature[node]]
    indices = torch.tensor([indices_row, indices_col], dtype=torch.long)
    values = [1/len(feature[node]) for node in range(n_nodes) for _ in range(len(feature[node]))]
    values = torch.FloatTensor(values)
    return indices, values