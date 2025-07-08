import torch
import torch.utils.data as data
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from data_utils.datasets_general_cf import PairwiseTrnData, AllRankTstData
from data_utils.datasets_diff import DiffusionData
import numpy as np
import scipy.sparse as sp
from config.configurator import configs
from os import path
from collections import defaultdict
from tqdm import tqdm
from .datasets_kg import KGTrainDataset, KGTestDataset, KGTripletDataset, generate_kg_batch
import random


class DataHandlerKG:
    """
    用于处理知识图谱（KG）相关数据的类，负责加载、处理和组织知识图谱与用户-物品交互数据，
    支持不同模型的数据加载需求，如扩散模型和普通模型。
    """
    def __init__(self) -> None:
        """
        初始化 DataHandlerKG 类，根据配置文件中的数据集名称设置数据文件路径，
        并初始化用户字典和实体数量等属性。
        """
        if configs['data']['name'] == 'mind':
            predir = './datasets/kg/mind_kg/'
        elif configs['data']['name'] == 'alibaba-fashion':
            predir = './datasets/kg/alibaba-fashion_kg/'
        elif configs['data']['name'] == 'last-fm':
            predir = './datasets/kg/last-fm_kg/'
        elif configs['data']['name'] == 'huggingface_2hop':
            predir = './datasets/kg/huggingface_2hop_kg/'
        elif configs['data']['name'] == 'huggingface_1hop':
            predir = './datasets/kg/huggingface_1hop_kg/'
        elif configs['data']['name'] == 'huggingface_homo':
            predir = './datasets/kg/huggingface_homo_kg/'
        elif configs['data']['name'] == 'huggingface_publish':
            predir = './datasets/kg/huggingface_publish_kg/'

        configs['data']['dir'] = predir
        self.trn_file = path.join(predir, 'train.txt')
        self.tst_file = path.join(predir, 'test.txt') 
        self.val_file = path.join(predir, 'valid.txt')
        self.kg_file = path.join(predir, 'kg_final.txt') 
        self.user_file = path.join(predir, 'user_list.txt')
        self.item_file = path.join(predir, 'item_list.txt') 
        self.entity_file = path.join(predir, 'entity_list.txt')
        self.relation_file = path.join(predir, 'relation_list.txt')
        # 读取实体列表文件，获取实体数量并加 1
        self.num_entities = np.genfromtxt(self.entity_file, dtype=None, names=True).shape[0] + 1
        self.train_user_dict = defaultdict(list)
        self.val_user_dict = defaultdict(list)
        self.test_user_dict = defaultdict(list)

    def _read_cf(self, file_name):
        """
        读取用户-物品交互数据文件，将每行数据解析为用户 ID 和物品 ID 对，去除重复项后返回。

        Args:
            file_name (str): 用户-物品交互数据文件的路径。

        Returns:
            np.ndarray: 包含用户 ID 和物品 ID 对的二维数组。
        """
        inter_mat = list()
        lines = open(file_name, "r").readlines()
        for l in lines:
            tmps = l.strip()
            inters = [int(i) for i in tmps.split(" ")]
            u_id, pos_ids = inters[0], inters[1:]
            pos_ids = list(set(pos_ids))
            for i_id in pos_ids:
                inter_mat.append([u_id, i_id])
        return np.array(inter_mat)
    
    def _read_cf_diff(self, file_name):
        """
        读取用于扩散模型的用户-物品交互数据文件，同时更新最大用户 ID 和最大物品 ID。

        Args:
            file_name (str): 用户-物品交互数据文件的路径。

        Returns:
            np.ndarray: 包含用户 ID 和物品 ID 对的二维数组。
        """
        inter_mat = list()
        lines = open(file_name, "r").readlines()
        for l in lines:
            tmps = l.strip()
            inters = [int(i) for i in tmps.split(" ")]
            u_id, pos_ids = inters[0], inters[1:]
            pos_ids = list(set(pos_ids))
            self.max_uid = max(self.max_uid, u_id)
            for i_id in pos_ids:
                inter_mat.append([u_id, i_id])
                self.max_iid = max(self.max_iid, i_id)
        return np.array(inter_mat)
    
    def _get_sp_mat(self, cf_data):
        """
        根据用户-物品交互数据构建稀疏矩阵。

        Args:
            cf_data (np.ndarray): 包含用户 ID 和物品 ID 对的二维数组。

        Returns:
            scipy.sparse.coo_matrix: 用户-物品交互的稀疏矩阵。
        """
        ui_edges = list()
        for u_id, i_id in cf_data:
            ui_edges.append([u_id, i_id])
        ui_edges = np.array(ui_edges)
        vals = [1.] * len(ui_edges)
        mat = sp.coo_matrix((vals, (ui_edges[:, 0], ui_edges[:, 1])), shape=(self.max_uid+1, self.max_iid+1))
        return mat
    
    def _read_triplets_diff(self, triplet_file, entity_num):
        """
        读取用于扩散模型的知识图谱三元组数据文件，生成正向和反向三元组，更新关系数量和实体数量配置。

        Args:
            triplet_file (str): 知识图谱三元组数据文件的路径。
            entity_num (int): 实体的数量。

        Returns:
            np.ndarray: 包含正向和反向三元组的二维数组。
        """
        can_triplets_np = np.loadtxt(triplet_file, dtype=np.int32)
        can_triplets_np = np.unique(can_triplets_np, axis=0)
        
        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)

        n_relations = max(triplets[:, 1]) + 1
        
        configs['data']['relation_num'] = n_relations
        configs['data']['entity_num'] = entity_num
        
        return triplets 
    
    def _collect_ui_dict(self, train_data, val_data, test_data):
        """
        收集训练集、验证集和测试集中的用户-物品交互数据，更新用户数量和物品数量配置，
        并填充用户字典。

        Args:
            train_data (np.ndarray): 训练集的用户-物品交互数据。
            val_data (np.ndarray): 验证集的用户-物品交互数据。
            test_data (np.ndarray): 测试集的用户-物品交互数据。
        """
        n_users = max(train_data[:, 0].max(), val_data[:, 0].max(), test_data[:, 0].max()) + 1
        n_items = max(train_data[:, 1].max(), val_data[:, 1].max(), test_data[:, 1].max()) + 1
        print("User num: %d, Item num: %d" % (n_users, n_items))
        configs['data']['user_num'] = n_users
        configs['data']['item_num'] = n_items

        for u_id, i_id in train_data:
            self.train_user_dict[int(u_id)].append(int(i_id))
        for u_id, i_id in test_data:
            self.test_user_dict[int(u_id)].append(int(i_id))
        for u_id, i_id in val_data:
            self.val_user_dict[int(u_id)].append(int(i_id))

    def _read_triplets(self, triplet_file, entity_num):
        """
        读取知识图谱三元组数据文件，生成正向和反向三元组，更新实体数量、节点数量、关系数量和三元组数量配置。

        Args:
            triplet_file (str): 知识图谱三元组数据文件的路径。
            entity_num (int): 实体的数量。

        Returns:
            np.ndarray: 包含正向和反向三元组的二维数组。
        """
        can_triplets_np = np.loadtxt(triplet_file, dtype=np.int32)
        can_triplets_np = np.unique(can_triplets_np, axis=0)

        # 获取反向三元组，例如 <entity, is-aspect-of, item>
        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
        # 考虑两个额外的关系 --- 'interact' 和 'be interacted'
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1
        # 获取完整版本的知识图谱
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)

        n_entities = entity_num
        n_nodes = n_entities + configs['data']['user_num']
        n_relations = max(triplets[:, 1]) + 1

        configs['data']['entity_num'] = n_entities
        configs['data']['node_num'] = n_nodes
        configs['data']['relation_num'] = n_relations
        configs['data']['triplet_num'] = len(triplets)

        return triplets

    def _build_graphs(self, train_data, triplets):
        """
        根据训练数据和知识图谱三元组构建知识图谱边、用户-物品交互边和知识图谱字典。

        Args:
            train_data (np.ndarray): 训练集的用户-物品交互数据。
            triplets (np.ndarray): 知识图谱三元组数据。

        Returns:
            tuple: 包含知识图谱边、用户-物品交互边和知识图谱字典的元组。
        """
        kg_dict = defaultdict(list)
        # h, t, r
        kg_edges = list()
        # u, i
        ui_edges = list()

        print("Begin to load interaction triples ...")
        for u_id, i_id in tqdm(train_data, ascii=True):
            ui_edges.append([u_id, i_id])

        print("Begin to load knowledge graph triples ...")
        for h_id, r_id, t_id in tqdm(triplets, ascii=True):
            # h,t,r
            kg_edges.append([h_id, t_id, r_id])
            kg_dict[h_id].append((r_id, t_id))

        return kg_edges, ui_edges, kg_dict

    def _build_graphs_diff(self, triplets):
        """
        根据知识图谱三元组构建用于扩散模型的知识图谱边和知识图谱字典，去除重复边。

        Args:
            triplets (np.ndarray): 知识图谱三元组数据。

        Returns:
            tuple: 包含知识图谱边和知识图谱字典的元组。
        """
        kg_dict = defaultdict(list)
        # h, t, r
        kg_edges = list()
        
        print("Begin to load knowledge graph triples ...")
        
        kg_counter_dict = {}
        
        for h_id, r_id, t_id in tqdm(triplets, ascii=True):
            if h_id not in kg_counter_dict.keys():
                kg_counter_dict[h_id] = set()
            if t_id not in kg_counter_dict[h_id]:
                kg_counter_dict[h_id].add(t_id)
            else:
                continue
            kg_edges.append([h_id, t_id, r_id])
            kg_dict[h_id].append((r_id, t_id))
            
        return kg_edges, kg_dict
    
    def buildKGMatrix(self, kg_edges):
        """
        根据知识图谱边构建知识图谱的稀疏矩阵。

        Args:
            kg_edges (list): 知识图谱边的列表。

        Returns:
            scipy.sparse.csr_matrix: 知识图谱的稀疏矩阵。
        """
        edge_list = []
        for h_id, t_id, r_id in kg_edges:
            edge_list.append((h_id, t_id))
        edge_list = np.array(edge_list)
        
        kgMatrix = sp.csr_matrix((np.ones_like(edge_list[:,0]), (edge_list[:,0], edge_list[:,1])), dtype='float64', shape=(configs['data']['entity_num'], configs['data']['entity_num']))
        
        return kgMatrix

    def _build_ui_mat(self, ui_edges):
        """
        根据用户-物品交互边构建用户-物品交互的稀疏矩阵。

        Args:
            ui_edges (list): 用户-物品交互边的列表。

        Returns:
            scipy.sparse.coo_matrix: 用户-物品交互的稀疏矩阵。
        """
        n_users = configs['data']['user_num']
        n_items = configs['data']['item_num']
        cf_edges = np.array(ui_edges)
        vals = [1.] * len(cf_edges)
        mat = sp.coo_matrix((vals, (cf_edges[:, 0], cf_edges[:, 1])), shape=(n_users, n_items))
        return mat
    
    def RelationDictBuild(self):
        """
        根据知识图谱字典构建关系字典，记录每个头实体到尾实体的关系。

        Returns:
            dict: 关系字典。
        """
        relation_dict = {}
        for head in self.kg_dict:
            relation_dict[head] = {}
            for (relation, tail) in self.kg_dict[head]:
                relation_dict[head][tail] = relation
        return relation_dict
    
    def buildUIMatrix(self, mat):
        """
        将用户-物品交互的稀疏矩阵转换为 PyTorch 稀疏张量，并移动到 GPU 上。

        Args:
            mat (scipy.sparse.coo_matrix): 用户-物品交互的稀疏矩阵。

        Returns:
            torch.sparse.FloatTensor: PyTorch 稀疏张量。
        """
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        return torch.sparse.FloatTensor(idxs, vals, shape).cuda()
    
    def _normalize_adj(self, mat):
        """
        对稀疏邻接矩阵进行拉普拉斯归一化。

        Args:
            mat (scipy.sparse.coo_matrix): 未归一化的邻接矩阵。

        Returns:
            scipy.sparse.coo_matrix: 归一化后的邻接矩阵。
        """
        # 添加一个小的常数以避免除零错误
        degree = np.array(mat.sum(axis=-1)) + 1e-10
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        return mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()

    def _make_torch_adj(self, mat):
        """
        将单向的稀疏邻接矩阵转换为双向的 PyTorch 稀疏张量。

        Args:
            mat (coo_matrix): 单向的邻接矩阵。

        Returns:
            torch.sparse.FloatTensor: 双向的矩阵。
        """
        a = csr_matrix((configs['data']['user_num'], configs['data']['user_num']))
        b = csr_matrix((configs['data']['item_num'], configs['data']['item_num']))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        # mat = (mat + sp.eye(mat.shape[0])) * 1.0# MARK
        mat = self._normalize_adj(mat)

        # 转换为 PyTorch 张量
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        return torch.sparse.FloatTensor(idxs, vals, shape).to(configs['device'])

    def load_data(self):
        """
        加载数据，根据配置文件中的模型类型选择不同的数据加载方式，
        初始化数据加载器和相关矩阵。
        """
        if 'diff_model' in configs['model'] and configs['model']['diff_model'] == 1:
            self.max_uid, self.max_iid = 0, 0
            trn_cf = self._read_cf_diff(self.trn_file)
            tst_cf = self._read_cf_diff(self.tst_file)
            val_cf = self._read_cf_diff(self.val_file)
            trn_mat = self._get_sp_mat(trn_cf)
            tst_mat = self._get_sp_mat(tst_cf)
            val_mat = self._get_sp_mat(val_cf)
            self.trn_mat = trn_mat
            configs['data']['user_num'], configs['data']['item_num'] = trn_mat.shape
            self.torch_adj = self._make_torch_adj(trn_mat)
            self.ui_matrix = self.buildUIMatrix(trn_mat)
            if configs['train']['loss'] == 'pairwise':
                trn_data = PairwiseTrnData(trn_mat)
            kg_triplets = self._read_triplets_diff(self.kg_file, self.num_entities)
            self.kg_edges, self.kg_dict = self._build_graphs_diff(kg_triplets)
            self.kg_matrix = self.buildKGMatrix(self.kg_edges)
            # 基于知识图谱矩阵初始化扩散数据
            self.diffusionData = DiffusionData(self.kg_matrix.A)
            self.diffusionLoader = data.DataLoader(self.diffusionData, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0)
            self.relation_dict = self.RelationDictBuild()
            val_data = AllRankTstData(val_mat, trn_mat)
            tst_data = AllRankTstData(tst_mat, trn_mat)
            self.valid_dataloader = data.DataLoader(val_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
            self.test_dataloader = data.DataLoader(tst_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
            self.train_dataloader = data.DataLoader(trn_data, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0)
        else:
            train_cf = self._read_cf(self.trn_file)
            test_cf = self._read_cf(self.tst_file)
            val_cf = self._read_cf(self.val_file)
            self._collect_ui_dict(train_cf, val_cf, test_cf)
            kg_triplets = self._read_triplets(self.kg_file, self.num_entities)
            self.kg_edges, self.ui_edges, self.kg_dict = self._build_graphs(train_cf, kg_triplets)
            self.ui_mat = self._build_ui_mat(self.ui_edges)

            val_data = KGTestDataset(self.val_user_dict, self.train_user_dict)
            self.valid_dataloader = data.DataLoader(val_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
            test_data = KGTestDataset(self.test_user_dict, self.train_user_dict)
            self.test_dataloader = data.DataLoader(test_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
            train_data = KGTrainDataset(train_cf, self.train_user_dict)
            self.train_dataloader = data.DataLoader(train_data, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0)

            if 'train_trans' in configs['model'] and configs['model']['train_trans']:
                triplet_data = KGTripletDataset(kg_triplets, self.kg_dict)
                # 不打乱数据，因为数据本身具有随机性
                self.triplet_dataloader = data.DataLoader(triplet_data, batch_size=configs['train']['kg_batch_size'], shuffle=False, num_workers=0)
    
    def generate_kg_batch(self):
        """
        生成知识图谱的批量数据。

        Returns:
            根据知识图谱字典和配置生成的批量数据。
        """
        return generate_kg_batch(self.kg_dict, configs['train']['kg_batch_size'], configs['data']['entity_num'])