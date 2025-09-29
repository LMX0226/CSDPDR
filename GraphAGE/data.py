
import random
import pandas as pd
import dgl
import networkx as nx
from sklearn.model_selection import StratifiedKFold
from utils import *
from sklearn.cluster import SpectralClustering
from scipy.sparse import coo_matrix

device = torch.device('cuda')

def get_adj(edges, size):
    edges_tensor = torch.LongTensor(edges).t()
    values = torch.ones(len(edges))
    adj = torch.sparse.LongTensor(edges_tensor, values, size).to_dense().long()
    adj = adj.numpy()
    return adj


def k_matrix(matrix, k):
    num = matrix.shape[0]
    knn_graph = np.zeros(matrix.shape)
    idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)
    for i in range(num):
        knn_graph[i, idx_sort[i, :k + 1]] = matrix[i, idx_sort[i, :k + 1]]
        knn_graph[idx_sort[i, :k + 1], i] = matrix[idx_sort[i, :k + 1], i]
    return knn_graph + np.eye(num)


def get_data(args):
    data = dict()

    drf = pd.read_csv(args.data_dir + 'DrugFingerprint.csv').iloc[:, 1:].to_numpy()
    drg = pd.read_csv(args.data_dir + 'DrugGIP.csv').iloc[:, 1:].to_numpy()

    dip = pd.read_csv(args.data_dir + 'DiseasePS.csv').iloc[:, 1:].to_numpy()
    dig = pd.read_csv(args.data_dir + 'DiseaseGIP.csv').iloc[:, 1:].to_numpy()

    prs = pd.read_csv(args.data_dir + 'Protein_sequence.csv').iloc[:, 1:].to_numpy()
    prg_r = pd.read_csv(args.data_dir + 'ProteinGIP_Drug.csv').iloc[:, 1:].to_numpy()
    prg_d = pd.read_csv(args.data_dir + 'ProteinGIP_Disease.csv').iloc[:, 1:].to_numpy()

    data['drug_number'] = int(drf.shape[0])
    data['disease_number'] = int(dig.shape[0])
    data['protein_number'] = int(prg_r.shape[0])

    data['drf'] = drf#药物F相似度
    data['drg'] = drg#药物G相似度
    data['dip'] = dip#diseaseP相似
    data['dig'] = dig#diseaseG相似
    data['prs'] = prs#proteinS相似
    data['prgr'] = prg_r
    data['prgd'] = prg_d

    #药物疾病关联[N,2]
    data['drdi'] = pd.read_csv(args.data_dir + 'DrugDiseaseAssociationNumber.csv', dtype=int).to_numpy()
    #药物蛋白关联[N,2]
    data['drpr'] = pd.read_csv(args.data_dir + 'DrugProteinAssociationNumber.csv', dtype=int).to_numpy()
    #蛋白疾病关联[N,2]
    data['dipr'] = pd.read_csv(args.data_dir + 'ProteinDiseaseAssociationNumber.csv', dtype=int).to_numpy()

    #[N,2]
    data['didr'] = data['drdi'][:, [1, 0]]
    data['prdr'] = data['drpr'][:, [1, 0]]
    data['prdi'] = data['dipr'][:, [1, 0]]

    drug_GAE = pd.read_csv(args.GAE_data_dir + 'drug.csv').iloc[:, 1:].to_numpy()
    disease_GAE = pd.read_csv(args.GAE_data_dir + 'disease.csv').iloc[:, 1:].to_numpy()
    protein_GAE = pd.read_csv(args.GAE_data_dir + 'protein.csv').iloc[:, 1:].to_numpy()
    data['drug_gae'] = drug_GAE
    data['disease_gae'] = disease_GAE
    data['protein_gae'] = protein_GAE

    return data


def data_processing(data, args):
    drdi_matrix = get_adj(data['drdi'], (args.drug_number, args.disease_number))
    one_index = []
    zero_index = []
    for i in range(drdi_matrix.shape[0]):
        for j in range(drdi_matrix.shape[1]):
            if drdi_matrix[i][j] >= 1:
                one_index.append([i, j])
            else:
                zero_index.append([i, j])
    random.seed(args.random_seed)
    random.shuffle(one_index)
    random.shuffle(zero_index)

    # unsamples=[],不进行采样的负样本
    unsamples = zero_index[int(args.negative_rate * len(one_index)):]
    data['unsample'] = np.array(unsamples)

    #作为采样的负样本
    zero_index = zero_index[:int(args.negative_rate * len(one_index))]

    index = np.array(one_index + zero_index, dtype=int)
    label = np.array([1] * len(one_index) + [0] * len(zero_index), dtype=int)
    samples = np.concatenate((index, np.expand_dims(label, axis=1)), axis=1)#[num_samples, 3]
    label_p = np.array([1] * len(one_index), dtype=int)#(len(one_index),)

    drdi_p = samples[samples[:, 2] == 1, :]#正样本
    drdi_n = samples[samples[:, 2] == 0, :]#负样本

    drs_mean = (data['drf'] + data['drg']) / 2
    dis_mean = (data['dip'] + data['dig']) / 2

    drs = np.where(data['drf'] == 0, data['drg'], drs_mean)
    dis = np.where(data['dip'] == 0, data['dip'], dis_mean)

    prg = (data['prgr'] + data['prgd']) / 2
    prs_mean = (data['prs'] + prg) / 2
    prs = np.where(data['prs'] == 0, prg, prs_mean)

    data['drs'] = drs
    data['dis'] = dis
    data['prs'] = prs
    data['all_samples'] = samples
    data['all_drdi'] = samples[:, :2]
    data['all_drdi_p'] = drdi_p
    data['all_drdi_n'] = drdi_n
    data['all_label'] = label
    data['all_label_p'] = label_p
    return data


def k_fold(data, args):
    k = args.k_fold
    skf = StratifiedKFold(n_splits=k, random_state=None, shuffle=False)
    X = data['all_drdi']
    Y = data['all_label']
    X_train_all, X_train_p_all, X_test_all, X_test_p_all, Y_train_all, Y_test_all = [], [], [], [], [], []
    X_train_n_all, X_test_n_all = [], []
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        Y_train = np.expand_dims(Y_train, axis=1).astype('float64')
        Y_test = np.expand_dims(Y_test, axis=1).astype('float64')
        X_train_p = X_train[Y_train[:, 0] == 1, :]
        X_train_n = X_train[Y_train[:, 0] == 0, :]
        X_test_p = X_test[Y_test[:, 0] == 1, :]
        X_test_n = X_test[Y_test[:, 0] == 0, :]
        X_train_all.append(X_train)
        X_train_p_all.append(X_train_p)
        X_train_n_all.append(X_train_n)
        X_test_all.append(X_test)
        X_test_p_all.append(X_test_p)
        X_test_n_all.append(X_test_n)
        Y_train_all.append(Y_train)
        Y_test_all.append(Y_test)

    data['X_train'] = X_train_all
    data['X_train_p'] = X_train_p_all
    data['X_train_n'] = X_train_n_all
    data['X_test'] = X_test_all
    data['X_test_p'] = X_test_p_all
    data['X_test_n'] = X_test_n_all
    data['Y_train'] = Y_train_all
    data['Y_test'] = Y_test_all
    return data

def dgl_similarity_graph(data, args):
    drdr_matrix = k_matrix(data['drs'], args.neighbor)
    didi_matrix = k_matrix(data['dis'], args.neighbor)
    prpr_matrix = k_matrix(data['prs'], args.neighbor)
    drdr_nx = nx.from_numpy_array(drdr_matrix)
    didi_nx = nx.from_numpy_array(didi_matrix)
    prpr_nx = nx.from_numpy_array(prpr_matrix)
    drdr_graph = dgl.from_networkx(drdr_nx)
    didi_graph = dgl.from_networkx(didi_nx)
    prpr_graph = dgl.from_networkx(prpr_nx)

    drdr_graph.ndata['drs'] = torch.tensor(data['drs'])
    didi_graph.ndata['dis'] = torch.tensor(data['dis'])
    prpr_graph.ndata['prs'] = torch.tensor(data['prs'])

    return drdr_graph, didi_graph, prpr_graph, data


def dgl_heterograph(data, edge, args):
    r_d = hetero_edge_index_to_adj_matrix(edge, args.drug_number, args.disease_number)
    d_r = r_d.T
    r_r = k_matrix(data['drs'], 10)
    d_d = k_matrix(data['dis'], 10)
    pr_pr = k_matrix(data['prs'], 40)
    r_pr = hetero_edge_index_to_adj_matrix(data['drpr'], args.drug_number, args.protein_number)
    pr_r = r_pr.T
    d_pr = hetero_edge_index_to_adj_matrix(data['dipr'], args.disease_number, args.protein_number)
    pr_d = d_pr.T

    # 定义每类节点的数量
    drug_number = args.drug_number
    disease_number = args.disease_number
    pr_number = args.protein_number

    # 初始化总体矩阵
    total_size = drug_number + disease_number + pr_number
    overall_adj_matrix = np.zeros((total_size, total_size))

    # 初始化边类型矩阵
    edge_type_matrix = np.zeros((total_size, total_size), dtype=int)

    # 定义边类型的代码
    EDGE_DRUG_DRUG = 1
    EDGE_DISEASE_DISEASE = 2
    EDGE_PR_PR = 3
    EDGE_DRUG_DISEASE = 4
    EDGE_DRUG_PR = 5
    EDGE_DISEASE_PR = 6
    EDGE_PPR_NEW = 7  # 新增 PPR 得到的边类型标记

    # 药物-药物 (Drug-Drug)
    overall_adj_matrix[:drug_number, :drug_number] = r_r
    edge_type_matrix[:drug_number, :drug_number][r_r != 0] = EDGE_DRUG_DRUG

    # 疾病-疾病 (Disease-Disease)
    overall_adj_matrix[drug_number:drug_number + disease_number, drug_number:drug_number + disease_number] = d_d
    edge_type_matrix[drug_number:drug_number + disease_number, drug_number:drug_number + disease_number][d_d != 0] = EDGE_DISEASE_DISEASE

    # PR-PR
    overall_adj_matrix[drug_number + disease_number:, drug_number + disease_number:] = pr_pr
    edge_type_matrix[drug_number + disease_number:, drug_number + disease_number:][pr_pr != 0] = EDGE_PR_PR

    # 药物-疾病 (Drug-Disease)
    overall_adj_matrix[:drug_number, drug_number:drug_number + disease_number] = r_d
    overall_adj_matrix[drug_number:drug_number + disease_number, :drug_number] = r_d.T
    edge_type_matrix[:drug_number, drug_number:drug_number + disease_number][r_d != 0] = EDGE_DRUG_DISEASE
    edge_type_matrix[drug_number:drug_number + disease_number, :drug_number][d_r != 0] = EDGE_DRUG_DISEASE

    # 药物-PR (Drug-PR)
    overall_adj_matrix[:drug_number, drug_number + disease_number:] = r_pr
    overall_adj_matrix[drug_number + disease_number:, :drug_number] = r_pr.T
    edge_type_matrix[:drug_number, drug_number + disease_number:][r_pr != 0] = EDGE_DRUG_PR
    edge_type_matrix[drug_number + disease_number:, :drug_number][pr_r != 0] = EDGE_DRUG_PR

    # 疾病-PR (Disease-PR)
    overall_adj_matrix[drug_number:drug_number + disease_number, drug_number + disease_number:] = d_pr
    overall_adj_matrix[drug_number + disease_number:, drug_number:drug_number + disease_number] = d_pr.T
    edge_type_matrix[drug_number:drug_number + disease_number, drug_number + disease_number:][d_pr != 0] = EDGE_DISEASE_PR
    edge_type_matrix[drug_number + disease_number:, drug_number:drug_number + disease_number][pr_d != 0] = EDGE_DISEASE_PR

    # 将 overall_adj_matrix 中所有非零元素替换为 1
    overall_adj_matrix[overall_adj_matrix != 0] = 1

    # 计算 PPR 扩散矩阵
    perturbed_adj_matrix = ppr_diffusion(overall_adj_matrix)
    perturbed_adj_matrix[perturbed_adj_matrix != 0] = 1

    # 创建原始矩阵和扩散后矩阵的边类型矩阵副本
    edge_type_overall = edge_type_matrix.copy()  # 原始矩阵的边类型
    edge_type_perturbed = edge_type_matrix.copy()  # 扩散后矩阵的边类型

    # 仅对 PPR 新增的边进行类型标记
    new_edges = (perturbed_adj_matrix > overall_adj_matrix).astype(int)

    # 清理 edge_type_perturbed 中未扩散新增的边的类型，保留新增边的标记
    edge_type_perturbed[new_edges == 1] = EDGE_PPR_NEW

    # 确保 edge_type_perturbed 中仅保留实际存在边的位置
    edge_type_perturbed[perturbed_adj_matrix == 0] = 0

    # 确保 edge_type_overall 中仅保留原始矩阵中实际存在边的位置
    edge_type_overall[overall_adj_matrix == 0] = 0

    # 统计非零元素数量
    nonzero_overall = np.count_nonzero(overall_adj_matrix)
    nonzero_perturbed = np.count_nonzero(perturbed_adj_matrix)

    nonzero_type_overall = np.count_nonzero(edge_type_overall)
    nonzero_type_perturbed = np.count_nonzero(edge_type_perturbed)
    # 将矩阵转换为 PyTorch 稀疏存储格式
    overall_adj_matrix_sparse = torch.sparse_coo_tensor(
        torch.tensor(overall_adj_matrix.nonzero(), dtype=torch.long),
        torch.tensor(overall_adj_matrix[overall_adj_matrix != 0], dtype=torch.float32),
        overall_adj_matrix.shape
    )

    perturbed_adj_matrix_sparse = torch.sparse_coo_tensor(
        torch.tensor(perturbed_adj_matrix.nonzero(), dtype=torch.long),
        torch.tensor(perturbed_adj_matrix[perturbed_adj_matrix != 0], dtype=torch.float32),
        perturbed_adj_matrix.shape
    )

    edge_type_overall_sparse = torch.sparse_coo_tensor(
        torch.tensor(edge_type_overall.nonzero(), dtype=torch.long),
        torch.tensor(edge_type_overall[edge_type_overall != 0], dtype=torch.int32),
        edge_type_overall.shape
    )

    edge_type_perturbed_sparse = torch.sparse_coo_tensor(
        torch.tensor(edge_type_perturbed.nonzero(), dtype=torch.long),
        torch.tensor(edge_type_perturbed[edge_type_perturbed != 0], dtype=torch.int32),
        edge_type_perturbed.shape
    )

    return (overall_adj_matrix_sparse, perturbed_adj_matrix_sparse,
            edge_type_overall_sparse, edge_type_perturbed_sparse)

def hetero_edge_index_to_adj_matrix(edge_index, num_A, num_B):

    # 初始化二部图邻接矩阵
    adj_matrix = np.zeros((num_A, num_B), dtype=int)

    # 遍历边索引，将每条边对应的类型 A 到类型 B 的位置设为 1
    for edge in edge_index:
        source, target = edge
        adj_matrix[source, target] = 1

    return adj_matrix

def partition_graph_with_spectral(original_adj_matrix, perturbed_adj_matrix, feature_matrix, edge_type_origin, edge_type_perturbed, K):
    """
    使用谱聚类算法将原始图和边干扰图划分为 K 个子图，优化为稀疏存储方式。

    参数：
        - original_adj_matrix: 原始图的邻接矩阵 (torch.sparse.FloatTensor, GPU 上)。
        - perturbed_adj_matrix: 边干扰图的邻接矩阵 (torch.sparse.FloatTensor, GPU 上)。
        - feature_matrix: 原图的特征矩阵 (torch.Tensor, GPU 上)。
        - edge_type_origin: 原始图的边类型矩阵 (torch.sparse.FloatTensor, GPU 上)。
        - edge_type_perturbed: 边干扰图的边类型矩阵 (torch.sparse.FloatTensor, GPU 上)。
        - K: 要划分的子图数量。

    返回：
        - original_subgraphs: 原始图子图信息的列表，每个子图为一个字典。
        - perturbed_subgraphs: 边干扰图子图信息的列表，结构与原始图子图列表相同。
    """
    # 转为稀疏矩阵的 COO 格式并移动到 CPU（SpectralClustering 需要）
    original_adj_coo = coo_matrix((original_adj_matrix.coalesce().values().cpu().numpy(),
                                   (original_adj_matrix.coalesce().indices()[0].cpu().numpy(),
                                    original_adj_matrix.coalesce().indices()[1].cpu().numpy())),
                                  shape=original_adj_matrix.shape)

    # 使用谱聚类进行节点划分
    spectral_clustering = SpectralClustering(n_clusters=K, affinity='precomputed', random_state=0)
    node_clusters = spectral_clustering.fit_predict(original_adj_coo)

    # 将节点聚类结果转为张量，并移动到 GPU
    node_clusters = torch.tensor(node_clusters, device=original_adj_matrix.device, dtype=torch.long)

    original_subgraphs = []
    perturbed_subgraphs = []

    for k in range(K):
        # 找到属于第 k 个子图的节点索引
        node_indices = (node_clusters == k).nonzero(as_tuple=False).squeeze()

        # 提取原始图的子图邻接矩阵
        sub_adj_matrix_original = original_adj_matrix.index_select(0, node_indices).index_select(1, node_indices)

        # 提取原始图的子图边类型矩阵
        sub_edge_types_original = edge_type_origin.index_select(0, node_indices).index_select(1, node_indices)

        # 提取原始图的特征矩阵
        sub_feature_matrix_original = feature_matrix.index_select(0, node_indices)

        # 提取边干扰图的子图邻接矩阵
        sub_adj_matrix_perturbed = perturbed_adj_matrix.index_select(0, node_indices).index_select(1, node_indices)

        # 提取边干扰图的子图边类型矩阵
        sub_edge_types_perturbed = edge_type_perturbed.index_select(0, node_indices).index_select(1, node_indices)

        # 提取边干扰图的特征矩阵
        sub_feature_matrix_perturbed = feature_matrix.index_select(0, node_indices)

        # 存储原始图的子图信息
        original_subgraphs.append({
            'sub_adj_matrix': sub_adj_matrix_original,
            'sub_feature_matrix': sub_feature_matrix_original,
            'sub_edge_types': sub_edge_types_original,
            'node_indices': node_indices
        })
        # 存储边干扰图的子图信息
        perturbed_subgraphs.append({
            'sub_adj_matrix': sub_adj_matrix_perturbed,
            'sub_feature_matrix': sub_feature_matrix_perturbed,
            'sub_edge_types': sub_edge_types_perturbed,
            'node_indices': node_indices
        })

    return original_subgraphs, perturbed_subgraphs




