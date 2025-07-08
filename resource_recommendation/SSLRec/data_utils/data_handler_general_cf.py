import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
import scipy.sparse as sp
from config.configurator import configs
from data_utils.datasets_general_cf import PairwiseTrnData, AllRankTstData, PairwiseWEpochFlagTrnData
import torch as t
import torch.utils.data as data

class DataHandlerGeneralCF:
    """
    用于处理通用协同过滤（General Collaborative Filtering）数据的类。
    该类负责加载、预处理和转换数据，以满足模型训练和测试的需求。
    """
    def __init__(self):
        """
        初始化 DataHandlerGeneralCF 类的实例，根据配置文件中的数据集名称设置数据文件路径。
        """
        if configs['data']['name'] == 'yelp':
            predir = './datasets/general_cf/sparse_yelp/'
        elif configs['data']['name'] == 'gowalla':
            predir = './datasets/general_cf/sparse_gowalla/'
        elif configs['data']['name'] == 'amazon':
            predir = './datasets/general_cf/sparse_amazon/'
        elif configs['data']['name'] == 'huggingface':
            predir = './datasets/general_cf/sparse_huggingface/'
        elif configs['data']['name'] == 'huggingface_model':
            predir = './datasets/general_cf/sparse_huggingface_model/'
        elif configs['data']['name'] == 'huggingface_dataset':
            predir = './datasets/general_cf/sparse_huggingface_dataset/'
        elif configs['data']['name'] == 'huggingface_space':
            predir = './datasets/general_cf/sparse_huggingface_space/'
        self.trn_file = predir + 'train_mat.pkl'  # 训练数据文件路径
        self.val_file = predir + 'valid_mat.pkl'  # 验证数据文件路径
        self.tst_file = predir + 'test_mat.pkl'   # 测试数据文件路径

    def _load_one_mat(self, file):
        """
        从指定文件中加载单个邻接矩阵。

        Args:
            file (string): 要加载的文件路径。

        Returns:
            scipy.sparse.coo_matrix: 加载的邻接矩阵。
        """
        with open(file, 'rb') as fs:
            mat = (pickle.load(fs) != 0).astype(np.float32)
        if type(mat) != coo_matrix:
            mat = coo_matrix(mat)
        return mat

    def _normalize_adj(self, mat):
        """
        对 coo_matrix 格式的矩阵进行拉普拉斯归一化。

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
        将 coo_matrix 格式的单向邻接矩阵转换为 torch.sparse.FloatTensor 格式的双向邻接矩阵。

        Args:
            mat (coo_matrix): 单向邻接矩阵。

        Returns:
            torch.sparse.FloatTensor: 双向邻接矩阵。
        """
        a = csr_matrix((configs['data']['user_num'], configs['data']['user_num']))
        b = csr_matrix((configs['data']['item_num'], configs['data']['item_num']))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        # mat = (mat + sp.eye(mat.shape[0])) * 1.0# MARK
        mat = self._normalize_adj(mat)

        # 转换为 PyTorch 稀疏张量
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).to(configs['device'])
    
    def load_data(self):
        """
        加载训练、验证和测试数据，对数据进行预处理，并创建数据加载器。
        """
        trn_mat = self._load_one_mat(self.trn_file)
        tst_mat = self._load_one_mat(self.tst_file)
        val_mat = self._load_one_mat(self.val_file)

        self.trn_mat = trn_mat
        configs['data']['user_num'], configs['data']['item_num'] = trn_mat.shape
        self.torch_adj = self._make_torch_adj(trn_mat)

        if configs['train']['loss'] == 'pairwise':
            trn_data = PairwiseTrnData(trn_mat)
        elif configs['train']['loss'] == 'pairwise_with_epoch_flag':
            trn_data = PairwiseWEpochFlagTrnData(trn_mat)
        # elif configs['train']['loss'] == 'pointwise':
        # 	trn_data = PointwiseTrnData(trn_mat)

        val_data = AllRankTstData(val_mat, trn_mat)
        tst_data = AllRankTstData(tst_mat, trn_mat)
        self.valid_dataloader = data.DataLoader(val_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
        self.test_dataloader = data.DataLoader(tst_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
        self.train_dataloader = data.DataLoader(trn_data, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0)