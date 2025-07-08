import torch
from cogdl.data import Graph
from cogdl.datasets import NodeDataset, generate_random_graph
from cogdl.experiments import experiment
from cogdl.utils import BCEWithLogitsLoss, MultiLabelMicroF1
from sklearn.preprocessing import StandardScaler

# 数据文件路径前缀
prefix = './data/'
# 数据集文件路径列表
paths = ['huggingface.pt']
# 要使用的模型列表
models = ['gcn']

# 自定义的 Huggingface 节点数据集类，继承自 NodeDataset
class HuggingfaceNodeDataset(NodeDataset):
    def __init__(self, path):
        # 保存数据集文件路径
        self.path = path
        # 调用父类构造函数，启用特征缩放，使用多标签 F1 分数作为评估指标
        super(HuggingfaceNodeDataset, self).__init__(path, scale_feat=True, metric="multilabel_f1")

# 自动化机器学习使用的搜索空间函数，用于定义超参数搜索范围
def search_space(trial):
    return {
        # 建议模型的层数，可选值为 2、3、4
        "num_layers": trial.suggest_categorical("num_layers", [2, 3, 4]),
        # 建议学习率，可选值为 1e-3、1e-4、1e-5，标记为配置项
        "lr": trial.suggest_categorical("lr", [1e-3, 1e-4, 1e-5]),  # config
        # 建议隐藏层大小，可选值为 256
        "hidden_size": trial.suggest_categorical("hidden_size", [256]),
        # 建议训练轮数，可选值为 500
        "epochs": trial.suggest_categorical("epochs", [500]),
        # 建议优化器，可选值为 "adamw"
        "optimizer": trial.suggest_categorical("optimizer", ["adamw"]),
        # 建议权重衰减系数，可选值为 0、1e-5、1e-4
        "weight_decay": trial.suggest_categorical("weight_decay", [0, 1e-5, 1e-4]),
        # 建议批量大小，可选值为 4096
        "batch_size": trial.suggest_categorical("batch_size", [4096]),
        # 建议早停耐心值，可选值为 100
        "patience": trial.suggest_categorical("patience", [100]),
    }

if __name__ == "__main__":
    # 通过定义新类来训练自定义数据集
    datasets = []
    for path in paths:
        # 创建 HuggingfaceNodeDataset 实例
        dataset = HuggingfaceNodeDataset(path=prefix + path)
        datasets.append(dataset)
    # 运行实验，指定数据集、模型、使用的设备、随机种子和超参数搜索空间
    experiment_result = experiment(dataset=datasets, model=models, devices=[0], seed=[0,1,2], search_space=search_space)
    
    # 这里可以根据 experiment_result 获取训练好的模型
    # 然后调用 classify_and_match 函数实现任务分类和工具匹配
    # 以下为伪代码示例
    # best_model = get_best_model(experiment_result)
    # matched_tools = classify_and_match(best_model, datasets[0])
    # print(matched_tools)