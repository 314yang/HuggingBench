# 工作

对 Hugging Face 构建知识图谱：
- 构建 AI 场景的要素，涉及模型（model ）、数据集（dataset ）、空间（space ，可理解为模型部署 / 展示环境 ）、集合（collection ，数据或模型的集合归类 ）、用户（user ）、组织（organization ）、任务（task ）、论文（paper ），是整个流程的 “输入” 。
- 梳理模型、数据、用户、机构等之间的复杂关联，让各要素关系更清晰：
    - 模型与数据：像 BERT（经典预训练模型 ）在 Wikipedia 数据集上训练（trained on ），medBERT 在 BookCorpus、PubMed 等数据上训练，体现模型训练的数据依赖；Embedding Models（嵌入模型 ）由 Google 等发布（publish ）、定义（define for ），BERT 会被论文引用（cite ） 。
    - 用户与模型：用户（如 Bob、Jack ）会有 “like（喜好 ）”“follow（关注 ）” 模型的行为，模型也可被用户微调（finetune ），体现人与模型的交互 。
知识图谱的应用功能，辅助资源利用、任务处理和模型管理。
- Resource Recommendation（资源推荐）：依据用户 “like（喜好 ）” 等行为，从数据库等资源里，给用户推荐模型、数据集等资源 。
- Task Classification（任务分类 ）：基于数据、模型，为任务（如具体 NLP 任务 ）做定义、分类，匹配合适工具。
- Model Tracing（模型溯源）：通过 “finetune（微调 ）” 等行为，追踪模型的迭代、衍生关系（比如一个模型微调后产生新模型，可追溯 lineage ）。

<img src='https://img.shields.io/badge/arXiv-2505.17507-b31b1b.svg'>


# 文件目录结构

```plaintext
HuggingBench/
├── LICENSE                   # 项目使用的 Apache 2.0 许可证文件
├── README.md                 # 项目的主说明文件，包含项目概述、数据链接、实验步骤等信息
├── figs/                     # 存放项目相关的图片，如 HuggingKG 图表
│   └── huggingkg.jpg         # HuggingKG 的图表文件
├── HuggingKG/                # 用于构建 HuggingKG 知识图谱的代码和文档
│   ├── HuggingKG_constructor.py  # 用于从 Hugging Face 爬取数据并构建知识图谱的主脚本
│   ├── requirements.txt      # 运行构建脚本所需的 Python 包列表
│   └── README.md             # 详细的构建过程和统计信息文档
├── resource_recommendation/  # 资源推荐任务相关代码
│   └── SSLRec/
│       ├── config/           # 配置文件目录
│       │   ├── configurator.py   # 解析命令行参数和 YAML 配置文件的脚本
│       │   └── modelconf/    # 模型配置文件目录
│       │       ├── bert4rec.yml    # BERT4Rec 模型的配置文件
│       │       ├── dccf.yml        # DCCF 模型的配置文件
│       │       ├── gformer.yml     # GFormer 模型的配置文件
│       │       ├── hccf.yml        # HCCF 模型的配置文件
│       │       ├── hmgcr.yml       # HMGCR 模型的配置文件
│       │       ├── lightgcl.yml    # LightGCL 模型的配置文件
│       │       ├── lightgcn.yml    # LightGCN 模型的配置文件
│       │       └── directau.yml    # DirectAU 模型的配置文件
│       ├── data_utils/       # 数据处理工具目录
│       │   ├── datasets_diff.py          # 差异数据集处理类
│       │   ├── datasets_general_cf.py    # 通用协同过滤数据集处理类
│       │   ├── datasets_sequential.py    # 序列数据集处理类
│       │   ├── datasets_multi_behavior.py# 多行为数据集处理类
│       │   ├── data_handler_sequential.py# 序列数据处理类
│       │   ├── data_handler_multi_behavior.py # 多行为数据处理类
│       │   ├── data_handler_kg.py        # 知识图谱数据处理类
│       │   ├── data_handler_social.py    # 社交数据处理类
│       │   ├── data_handler_general_cf.py# 通用协同过滤数据处理类
│       │   └── __init__.py       # 包初始化文件
│       └── scripts/        # 实验脚本目录
│           ├── run_general_cf.sh    # 运行通用协同过滤 collaborative filtering 实验的脚本 
│           ├── run_social_rec.sh    # 运行社交推荐 social recommendation 实验的脚本
│           ├── run_kg_based_rec.sh  # 运行基于知识图谱推荐 HuggingKG recommendation 实验的脚本
├── task_classification/      # 任务分类任务相关代码
│   └── tune_huggingface.py   # 任务分类实验的调优脚本
└── model_tracing/            # 模型追踪任务相关代码
    ├── kge/
    │   ├── scripts/
    │   │   ├── train.sh      # 模型训练脚本
    │   │   └── test.sh       # 模型测试脚本
    │   └── examples/
    │       ├── huggingface-train-complex.yaml    # Complex 模型训练配置文件
    │       ├── huggingface-train-rotate.yaml     # Rotate 模型训练配置文件
    │       ├── huggingface-train-transe.yaml     # TransE 模型训练配置文件
    │       └── huggingface-train-transformer.yaml# Transformer 模型训练配置文件
```


### 主要任务
该项目围绕Hugging Face知识图谱开展了资源推荐、任务分类和模型追踪三个主要任务的基准测试，了解各个模型在不同指标下的性能表现，从而为模型的选择和优化提供参考依据。 

### 具体操作及结果

#### 1. HuggingKG知识图谱构建
- **操作**：使用`HuggingKG_constructor.py`脚本从Hugging Face爬取数据并构建知识图谱。脚本中使用了多线程加速数据处理，通过`run`方法顺序执行数据收集、验证和存储操作。具体的数据来源和处理方式如下：
    - **实体**：从不同的API端点和数据标签中获取任务、模型、数据集、空间、集合、论文、用户和组织等实体信息。
    - **关系**：根据模型、数据集等数据中的标签信息，提取模型与任务、模型与模型、模型与数据集等之间的关系。
- **结果**：构建了一个名为HuggingKG的综合知识图谱，包含丰富的节点和边信息。知识图谱的数据以JSON文件的形式存储，包括实体数据、关系数据和额外的ID集合数据。可在 [Hugging Face](https://huggingface.co/datasets/cqsss/HuggingKG) 上获取，其中`triples.txt`包含图三元组集合，`HuggingKG_V20241215174821.zip`包含详细节点和边属性的JSON文件。

```plaintext
# triples.txt

# 节点 ->边-> 节点
# JeffreyXiang/TRELLIS空间 使用了 JeffreyXiang/TRELLIS-image-large模型
JeffreyXiang/TRELLIS space_use_model JeffreyXiang/TRELLIS-image-large

# Changg/ori 模型 用于 “文本到图像” 任务
Changg/ori model_definedFor_task text-to-image

```

#### 2. 资源推荐
- **操作**：使用 [SSLRec](https://github.com/HKUDS/SSLRec) 实现资源推荐任务。具体步骤如下：
    1. 克隆`SSLRec`并安装依赖。
    2. 将HuggingBench数据放置在`SSLRec/datasets`目录下。
    3. 复制配置文件`./resource_recommendation/SSLRec/config`。
    4. 运行`./resource_recommendation/SSLRec/scripts`中的脚本。
- **结果**：在通用协同过滤、社交推荐和基于知识图谱的推荐任务上，评估了多种算法（如LightGCN、HCCF、SimGCL等）的性能，记录了不同算法在Recall和NDCG指标下的表现。

测试指标
 - **Recall@k**：召回率，指在真实相关的项目中，被模型推荐到前 `k` 个位置的项目比例。例如，`Recall@5` 表示真实相关的项目中，有多少比例被模型推荐到了前 5 个位置。该指标值越高，说明模型能召回更多相关的项目。
 - **NDCG@k（Normalized Discounted Cumulative Gain）**：归一化折损累积增益，综合考虑了推荐结果的相关性和排名顺序，数值越高表示推荐结果越相关且排名越合理。

以“General Collaborative Filtering”部分为例：
|  | Recall@5 | Recall@10 | Recall@20 | Recall@40 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@40 |
|--|--|--|--|--|--|--|--|--|
| LightGCN | 0.0856 | 0.1301 | 0.1932 | 0.2759 | 0.0868 | 0.1003 | 0.1192 | 0.1413 |
| HCCF | 0.0834 | 0.1254 | 0.1820 | 0.2504 | 0.0847 | 0.0975 | 0.1143 | 0.1328 |
| SimGCL | 0.0999 | 0.1515 | 0.2186 | 0.3010 | 0.0998 | 0.1158 | 0.1358 | 0.1581 |
| LightGCL | 0.1033 | 0.1558 | 0.2228 | 0.3017 | 0.1035 | 0.1198 | 0.1398 | 0.1611 |
| AutoCF | 0.1003 | 0.1530 | 0.2190 | 0.3039 | 0.1012 | 0.1174 | 0.1371 | 0.1598 |
| DCCF | 0.0985 | 0.1493 | 0.2167 | 0.3003 | 0.0983 | 0.1142 | 0.1343 | 0.1567 |

从表格数据可以看出，`LightGCL` 模型在 `Recall@k` 和 `NDCG@k` 指标上的表现相对较好，说明该模型在召回相关项目和提供合理推荐排名方面具有优势。而 `HCCF` 模型的各项指标值相对较低，性能相对较弱。

#### 3. 任务分类
- **操作**：使用 [CogDL](https://github.com/THUDM/CogDL) 实现任务分类任务。具体步骤如下：
    1. 克隆`CogDL`并安装依赖。
    2. 下载 [HuggingBench-Classification](https://huggingface.co/datasets/cqsss/HuggingBench-Classification) 数据到`task_classification/data/`目录。
    3. 运行`./task_classification/tune_huggingface.py`。
- **结果**：评估了多种图神经网络模型（如GCN、GAT、GRAND等）在任务分类任务上的性能，记录了不同模型在不同特征表示（如binary、BERT、BGE等）下的表现。


每一行代表一个不同的图神经网络模型，这些模型被用于任务分类任务，具体解释如下：
- **GCN（Graph Convolutional Network）**：图卷积网络，是一种经典的图神经网络，通过聚合节点的邻域信息来学习节点的表示。
- **GAT（Graph Attention Network）**：图注意力网络，引入了注意力机制，能够自适应地为不同邻域节点分配不同的权重。
- **GRAND（Graph Random Neural Network）**：一种基于随机游走的图神经网络，通过随机采样和聚合邻域信息来学习节点表示。
- **GraphSAGE（Graph Sample and Aggregate）**：图采样与聚合网络，通过采样和聚合邻域节点的信息来学习节点表示，适用于大规模图数据。
- **ANNPN（Adaptive Neural Network Propagation Network）**：自适应神经网络传播网络，可能是一种自定义的图神经网络模型，用于处理特定的图数据和任务。
- **GCNII（Graph Convolutional Network II）**：可能是 GCN 的改进版本，通过引入额外的层和机制来提高模型的性能。
- **GraphSAINT（Graph Sampling Based Inductive Learning Method）**：基于图采样的归纳学习方法，通过采样子图来训练模型，适用于大规模图数据。
- **RevGCN（Reversible Graph Convolutional Network）**：可逆图卷积网络，引入了可逆模块，能够减少内存消耗和提高训练效率。
- **RevGAT（Reversible Graph Attention Network）**：可逆图注意力网络，结合了可逆模块和注意力机制，用于处理图数据。

每一列代表一种不同的特征表示方法，用于为模型提供输入数据，具体解释如下：
- **binary**：二进制特征表示，可能是将数据转换为二进制向量作为输入。
- **BERT**：使用预训练的 BERT 模型提取的特征，BERT 是一种基于 Transformer 的预训练语言模型，能够学习到丰富的语义信息。
- **BERT (ft)**：在预训练的 BERT 模型基础上进行微调（fine-tuning）后提取的特征，微调可以使模型更好地适应特定的任务。
- **BGE**：可能是一种自定义的特征提取方法，用于提取图数据的特征。“Bidirectional Graph Embedding” 双向图嵌入
- **BGE (ft)**：在 BGE 特征提取方法的基础上进行微调后提取的特征。 “fine-tuned”（微调）

|            | binary  | BERT    | BERT (ft) | BGE     | BGE (ft)  |
|------------|---------|---------|---------|---------|---------|
| GCN        | 0.0662  | 0.7620  | 0.8291  | 0.7411  | 0.8522  |
| GAT        | 0.0390  | 0.5105  | 0.8125  | 0.5444  | 0.8261  |
| GRAND      | 0.1228  | 0.1297  | 0.6089  | 0.2646  | 0.4532  |
| GraphSAGE  | 0.1800  | 0.5341  | 0.8845  | 0.8199  | 0.8830  |
| ANNPN      | 0.0448  | 0.7297  | 0.8304  | 0.7571  | 0.8419  |
| GCNII      | 0.1149  | 0.6456  | 0.8836  | 0.7779  | 0.8802  |
| GraphSAINT | 0.0579  | 0.2703  | 0.8342  | 0.0540  | 0.8251  |
| RevGCN     | 0.1071  | 0.6763  | 0.8851  | 0.8039  | 0.8770  |
| RevGAT     | 0.0335  | 0.7412  | 0.8849  | 0.7569  | 0.8716  |

从表格数据可知，`GraphSAGE` 模型在 `BERT (ft)` 和 `BGE` 特征表示下的分类准确率较高，说明该模型在这两种特征下的分类性能较好。而 `GRAND` 模型在部分特征表示下的准确率较低，性能相对较差。



#### 4. 模型溯源
- **操作**：使用 [LibKGE](https://github.com/uma-pi1/kge) 知识图嵌入（Knowledge Graph Embedding，KGE）实现有监督基线；使用 [ULTRA](https://github.com/DeepGraphLearning/ULTRA) 和 [KG-ICL](https://github.com/nju-websoft/KG-ICL) 的官方代码实现两个无监督模型。具体步骤如下：
    1. 克隆`LibKGE`并安装依赖。
    2. 下载 [HuggingBench-Tracing](https://huggingface.co/datasets/cqsss/HuggingBench-Tracing) 数据到`kge/data/huggingface`。
    3. 复制配置文件`./model_tracing/kge/examples`。
    4. 运行训练/测试脚本`model_tracing\kge\scripts\train.sh`和`model_tracing\kge\scripts\test.sh`。
- **结果**：评估了多种模型（如RESCAL、TransE、DistMult等）在模型追踪任务上的性能，记录了不同模型在MRR和HIT@k指标下的表现。


测试指标
 - **MRR（Mean Reciprocal Rank）**：平均倒数排名，反映了模型预测结果中正确答案的排名情况，数值越高表明模型预测的结果越靠前，性能越好。
 - **HIT@k**：指在模型预测的前 `k` 个结果中，包含正确答案的比例。例如，`HIT@1` 表示第一个预测结果就是正确答案的比例，`HIT@3` 表示前三个预测结果中包含正确答案的比例，以此类推。该指标值越高，说明模型的命中率越高。


|  | MRR | HIT@1 | HIT@3 | HIT@5 | HIT@10 |
|--|--|--|--|--|--|
| RESCAL | 0.2694 | 0.2380 | 0.2667 | 0.2929 | 0.3470 |
| TransE | 0.5589 | 0.4496 | 0.6321 | 0.6973 | 0.7562 |
| DistMult | 0.2050 | 0.1421 | 0.2321 | 0.2735 | 0.3324 |
| ComplEx | 0.1807 | 0.1109 | 0.2122 | 0.2599 | 0.3066 |
| ConvE | 0.4739 | 0.3766 | 0.5119 | 0.5903 | 0.6735 |
| RotatE | 0.5317 | 0.4195 | 0.6029 | 0.6803 | 0.7392 |
| HittER | 0.3678 | 0.2900 | 0.4078 | 0.4657 | 0.5314 |
| ULTRA(无监督模型) | 0.3373 | 0.1440 | 0.4803 | 0.5309 | 0.6672 |
| KG - ICL(无监督模型) | 0.4008 | 0.3354 | 0.3792 | 0.4854 | 0.5938 |

从表格数据可知，`TransE` 模型在各项指标上的表现相对较好，尤其是 `MRR` 和 `HIT@k` 指标值都比较高，这表明 `TransE` 模型在预测时能将正确答案排在较前的位置，且命中率较高。而 `ComplEx` 模型的各项指标值相对较低，说明其性能在这些模型中相对较差。

