import torch
from cogdl.data import Graph
from cogdl.datasets import NodeDataset, generate_random_graph
from cogdl.experiments import experiment
from cogdl.utils import BCEWithLogitsLoss, MultiLabelMicroF1
from sklearn.preprocessing import StandardScaler

prefix = './data/'
paths = ['huggingface.pt']
models = ['gcn']

class HuggingfaceNodeDataset(NodeDataset):
    def __init__(self, path):
        self.path = path
        super(HuggingfaceNodeDataset, self).__init__(path, scale_feat=True, metric="multilabel_f1")

# automl usage
def search_space(trial):
    return {
        "num_layers": trial.suggest_categorical("num_layers", [2, 3, 4]),
        "lr": trial.suggest_categorical("lr", [1e-3, 1e-4, 1e-5]),  # config
        "hidden_size": trial.suggest_categorical("hidden_size", [256]),
        "epochs": trial.suggest_categorical("epochs", [500]),
        "optimizer": trial.suggest_categorical("optimizer", ["adamw"]),
        "weight_decay": trial.suggest_categorical("weight_decay", [0, 1e-5, 1e-4]),
        "batch_size": trial.suggest_categorical("batch_size", [4096]),
        "patience": trial.suggest_categorical("patience", [100]),
    }

if __name__ == "__main__":
    # Train customized dataset via defining a new class
    datasets = []
    for path in paths:
        dataset = HuggingfaceNodeDataset(path=prefix + path)
        datasets.append(dataset)
    experiment(dataset=datasets, model=models, devices=[0], seed=[0,1,2], search_space=search_space)