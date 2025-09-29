import numpy as np
import scipy.sparse as sp
import torch

def ppr_diffusion(adj_matrix, alpha=0.1, tol=1e-4, threshold=1e-3):
    """
    使用 PPR 扩散对邻接矩阵进行稀疏扩散，并返回二维数组格式的结果。

    参数：
        - adj_matrix: 原始邻接矩阵 (numpy 2D array)。
        - alpha: 扩散系数，通常取值在 (0, 1)。
        - tol: 逆矩阵计算时的容忍误差。
        - threshold: 用于稀疏化的阈值，小于该值的元素将被置零。

    返回：
        - ppr_matrix: 稀疏化的 PPR 扩散矩阵 (numpy 2D array)。
    """
    # 将输入邻接矩阵转换为稀疏矩阵格式
    adj_matrix = sp.csr_matrix(adj_matrix)

    # 计算度矩阵 D 的逆
    deg = np.array(adj_matrix.sum(axis=1)).flatten()  # 每个节点的度
    deg_inv = sp.diags(1.0 / deg)  # 度的倒数作为对角线矩阵

    # 计算扩散矩阵 S = alpha * (I - (1 - alpha) * A * D^-1)^-1
    identity = sp.eye(adj_matrix.shape[0])
    transition_matrix = (1 - alpha) * adj_matrix.dot(deg_inv)
    diff_matrix = sp.linalg.inv(identity - transition_matrix).tocsc()
    ppr_matrix = alpha * diff_matrix

    # 稀疏化处理：小于阈值的元素置零，并转换为二维数组
    ppr_matrix[ppr_matrix < threshold] = 0
    ppr_matrix = ppr_matrix.toarray()  # 转换为 numpy 2D array 格式

    return ppr_matrix


def restore_original_feature_matrix(subgraphs_feat, node_indices, original_num_nodes):
    """
    根据子图特征和节点索引还原原图的特征矩阵。

    参数：
        - subgraphs_feat: 包含每个子图特征矩阵的列表 (list of torch.Tensors)。
        - node_indices: 包含每个子图中节点索引的列表。
        - original_num_nodes: 原图的节点总数。
        - device: 设备 (torch.device)，原图特征矩阵要存储的设备。

    返回：
        - original_feature_matrix: 还原后的原图特征矩阵 (torch.Tensor)。
    """
    # 初始化一个空的原图特征矩阵，确保在指定设备上
    feature_dim = subgraphs_feat[0].shape[1]
    deveice = subgraphs_feat[0].device

    original_feature_matrix = torch.zeros((original_num_nodes, feature_dim)).to(deveice)

    # 遍历每个子图，将子图特征矩阵放回原图的相应位置
    for i, sub_feature_matrix in enumerate(subgraphs_feat):
        nodes_in_subgraph = node_indices[i]

        # 直接将子图特征矩阵插入到原图对应位置
        original_feature_matrix.index_copy_(0, nodes_in_subgraph, sub_feature_matrix)

    return original_feature_matrix