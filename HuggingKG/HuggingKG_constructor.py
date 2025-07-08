import concurrent.futures
import json
import logging
import os
from datetime import datetime

import requests
from huggingface_hub import (get_dataset_tags, get_model_tags, list_datasets,
                             list_models, list_spaces, login, logout)
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3 import Retry

# 定义知识图谱构造器类
class KGConstructor:
    def __init__(self):
        # 设置环境，包括登录、创建输出目录和配置日志
        self.setup_environment()
        # 初始化数据结构，用于存储实体、关系和额外数据
        self.init_data_structures()
        # 设置请求会话，包含重试机制
        self.setup_session()

    def setup_environment(self):
        """Setup environment variables and logging"""
        # 请添加你的 Hugging Face API 令牌
        self.my_hf_token = "" 
        # 使用提供的令牌登录 Hugging Face
        login(token=self.my_hf_token)
        # 获取当前时间戳，用于创建唯一的输出目录
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # 创建以时间戳命名的输出目录
        self.output_dir = "HuggingKG_V"+timestamp
        os.makedirs(self.output_dir, exist_ok=True)
        # 配置日志，将日志信息写入输出目录下的 logs.log 文件
        logging.basicConfig(
            filename=os.path.join(self.output_dir, 'logs.log'), 
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def setup_session(self):
        """Setup requests session with retry mechanism"""
        # 创建一个请求会话对象
        self.session = requests.Session()
        # 配置重试机制
        retries = Retry(
            total=10, 
            backoff_factor=1, 
            # 遇到这些状态码时进行重试
            status_forcelist=[408, 429, 502, 503, 504, 522, 524],
            # 允许重试的请求方法
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE"],
            # 尊重服务器的重试等待时间头信息
            respect_retry_after_header=True
            )
        # 创建一个 HTTP 适配器，设置最大重试次数、连接池和最大连接数
        adapter = HTTPAdapter(
            max_retries=retries,
            pool_connections=200,
            pool_maxsize=200,
            pool_block=True
            )
        # 将适配器挂载到 HTTP 和 HTTPS 请求上
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)
        # 更新会话的请求头，包括用户代理和授权信息
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
            'Authorization': f"Bearer {self.my_hf_token if self.my_hf_token else ''}"
        })

    def init_data_structures(self):
        # entity data
        # 存储处理后的任务信息
        self.processed_tasks = []
        # 存储处理后的模型信息
        self.processed_models = []
        # 存储处理后的数据集信息
        self.processed_datasets = []
        # 存储处理后的空间信息
        self.processed_spaces = []
        # 存储处理后的论文信息
        self.processed_papers = []
        # 存储处理后的集合信息
        self.processed_collections = []
        # 存储处理后的用户信息
        self.processed_users = []
        # 存储处理后的组织信息
        self.processed_orgs = []

        # relation data
        # 存储模型与任务的定义关系
        self.model_definedFor_task = []
        # 存储模型的适配器关系
        self.model_adapter_model = []
        # 存储模型的微调关系
        self.model_finetune_model = []
        # 存储模型的合并关系
        self.model_merge_model = []
        # 存储模型的量化关系
        self.model_quantized_model = []
        # 存储模型与数据集的训练或微调关系
        self.model_trainedOrFineTunedOn_dataset = []
        # 存储模型与论文的引用关系
        self.model_cite_paper = []
        # 存储数据集与任务的定义关系
        self.dataset_definedFor_task = []
        # 存储数据集与论文的引用关系
        self.dataset_cite_paper = []
        # 存储空间与模型的使用关系
        self.space_use_model = []
        # 存储空间与数据集的使用关系
        self.space_use_dataset = []
        # 存储集合与模型的包含关系
        self.collection_contain_model = []
        # 存储集合与数据集的包含关系
        self.collection_contain_dataset = []
        # 存储集合与空间的包含关系
        self.collection_contain_space = []
        # 存储集合与论文的包含关系
        self.collection_contain_paper = []
        # 存储用户名与模型的发布关系
        self.username_publish_model = []
        # 存储用户名与数据集的发布关系
        self.username_publish_dataset = []
        # 存储用户名与空间的发布关系
        self.username_publish_space = []
        # 存储用户与模型的发布关系
        self.user_publish_model = []
        # 存储用户与数据集的发布关系
        self.user_publish_dataset = []
        # 存储用户与空间的发布关系
        self.user_publish_space = []
        # 存储用户与论文的发布关系
        self.user_publish_paper = []
        # 存储用户与集合的拥有关系
        self.user_own_collection = []
        # 存储用户与模型的喜欢关系
        self.user_like_model = []
        # 存储用户与数据集的喜欢关系
        self.user_like_dataset = []
        # 存储用户与空间的喜欢关系
        self.user_like_space = []
        # 存储用户与用户的关注关系
        self.user_follow_user = []
        # 存储用户与组织的附属关系
        self.user_affiliatedWith_org = []
        # 存储用户与组织的关注关系
        self.user_follow_org = []
        # 存储组织与模型的发布关系
        self.org_publish_model = []
        # 存储组织与数据集的发布关系
        self.org_publish_dataset = []
        # 存储组织与空间的发布关系
        self.org_publish_space = []
        # 存储组织与集合的拥有关系
        self.org_own_collection = []

        # extra data
        # 存储任务的 ID 集合
        self.task_ids = set()
        # 存储模型的 ID 集合
        self.model_ids = set()
        # 存储数据集的 ID 集合
        self.dataset_ids = set()
        # 存储空间的 ID 集合
        self.space_ids = set()
        # 存储论文的 ID 集合
        self.paper_ids = set()
        # 存储集合的 slug 集合
        self.collection_slugs = set()
        # 存储用户的 ID 集合
        self.user_ids = set()
        # 存储组织的 ID 集合
        self.org_ids = set()

        # 存储 arXiv ID 集合
        self.arxiv_ids = set()
        # 存储用户名 ID 集合
        self.username_ids = set()
        # self.visited_user_ids = set()
        # self.visited_org_ids = set()

    def save_data(self, filename, data):
        """Helper method to save data to JSON file"""
        # 将数据保存为 JSON 文件，使用指定的文件名和输出目录
        with open(os.path.join(self.output_dir, f'{filename}.json'), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    def save_entity_data(self):
        print("Saving entity data...")
        # 定义要保存的实体数据列表
        save_entities = [
            ('tasks', self.processed_tasks),
            ('models', self.processed_models),
            ('datasets', self.processed_datasets),
            ('spaces', self.processed_spaces),
            ('papers', self.processed_papers),
            ('collections', self.processed_collections),
            ('users', self.processed_users),
            ('orgs', self.processed_orgs)
        ]
        # 遍历实体数据列表，调用 save_data 方法保存每个实体数据
        for entity_name, data in save_entities:
            self.save_data(entity_name, data)

    def save_relation_data(self):
        print("Saving relation data...")
        # 定义要保存的关系数据列表
        save_relations = [
            ('model_definedFor_task', self.model_definedFor_task),
            ('model_adapter_model', self.model_adapter_model),
            ('model_finetune_model', self.model_finetune_model),
            ('model_merge_model', self.model_merge_model),
            ('model_quantized_model', self.model_quantized_model),
            ('model_trainedOrFineTunedOn_dataset', self.model_trainedOrFineTunedOn_dataset),
            ('model_cite_paper', self.model_cite_paper),
            ('dataset_definedFor_task', self.dataset_definedFor_task),
            ('dataset_cite_paper', self.dataset_cite_paper),
            ('space_use_model', self.space_use_model),
            ('space_use_dataset', self.space_use_dataset),
            ('collection_contain_model', self.collection_contain_model),
            ('collection_contain_dataset', self.collection_contain_dataset),
            ('collection_contain_space', self.collection_contain_space),
            ('collection_contain_paper', self.collection_contain_paper),
            ('user_publish_model', self.user_publish_model),
            ('user_publish_dataset', self.user_publish_dataset),
            ('user_publish_space', self.user_publish_space),
            ('user_publish_paper', self.user_publish_paper),
            ('user_own_collection', self.user_own_collection),
            ('user_like_model', self.user_like_model),
            ('user_like_dataset', self.user_like_dataset),
            ('user_like_space', self.user_like_space),
            ('user_follow_user', self.user_follow_user),
            ('user_affiliatedWith_org', self.user_affiliatedWith_org),
            ('user_follow_org', self.user_follow_org),
            ('org_publish_model', self.org_publish_model),
            ('org_publish_dataset', self.org_publish_dataset),
            ('org_publish_space', self.org_publish_space),
            ('org_own_collection', self.org_own_collection)
        ]
        # 遍历关系数据列表，调用 save_data 方法保存每个关系数据
        for relation_name, data in save_relations:
            self.save_data(relation_name, data)

    def save_extra_data(self):
        print("Saving extra data...")
        # 定义要保存的额外数据列表
        save_extra_data = [
            # 'extra_data_name', data
            ('task_ids', self.task_ids),
            ('model_ids', self.model_ids),
            ('dataset_ids', self.dataset_ids),
            ('space_ids', self.space_ids),
            ('paper_ids', self.paper_ids),
            ('collection_slugs', self.collection_slugs),
            ('user_ids', self.user_ids),
            ('org_ids', self.org_ids),

            ('arxiv_ids', self.arxiv_ids),
            ('username_ids', self.username_ids),
            ('username_publish_model', self.username_publish_model),
            ('username_publish_dataset', self.username_publish_dataset),
            ('username_publish_space', self.username_publish_space)
        ]
        
        # 在输出目录下创建 extra_data 子目录
        os.makedirs(os.path.join(self.output_dir, 'extra_data'), exist_ok=True)

        # 遍历额外数据列表，将数据保存到 extra_data 子目录下的 JSON 文件中
        for extra_data_name, data in save_extra_data:
            with open(os.path.join(self.output_dir, 'extra_data', f'{extra_data_name}.json'), 'w') as f:
                json.dump(data, f, indent=4)

    def save_all_data(self):
        # print("Saving all data...")
        # 保存实体数据
        self.save_entity_data()
        # 保存关系数据
        self.save_relation_data()

    def log_stats(self):
        """Log statistics about processed data"""

        # 统计每个实体类型的数量
        entity_stats = {
            'tasks': len(self.processed_tasks),
            'models': len(self.processed_models),
            'datasets': len(self.processed_datasets),
            'spaces': len(self.processed_spaces),
            'papers': len(self.processed_papers),
            'collections': len(self.processed_collections),
            'users': len(self.processed_users),
            'orgs': len(self.processed_orgs)
        }

        # 统计每个关系类型的数量
        relation_stats = {
            'model_definedFor_task': len(self.model_definedFor_task),
            'model_adapter_model': len(self.model_adapter_model),
            'model_finetune_model': len(self.model_finetune_model),
            'model_merge_model': len(self.model_merge_model),
            'model_quantized_model': len(self.model_quantized_model),
            'model_trainedOrFineTunedOn_dataset': len(self.model_trainedOrFineTunedOn_dataset),
            'model_cite_paper': len(self.model_cite_paper),
            'dataset_definedFor_task': len(self.dataset_definedFor_task),
            'dataset_cite_paper': len(self.dataset_cite_paper),
            'space_use_model': len(self.space_use_model),
            'space_use_dataset': len(self.space_use_dataset),
            'collection_contain_model': len(self.collection_contain_model),
            'collection_contain_dataset': len(self.collection_contain_dataset),
            'collection_contain_space': len(self.collection_contain_space),
            'collection_contain_paper': len(self.collection_contain_paper),
            'user_publish_model': len(self.user_publish_model),
            'user_publish_dataset': len(self.user_publish_dataset),
            'user_publish_space': len(self.user_publish_space),
            'user_publish_paper': len(self.user_publish_paper),
            'user_own_collection': len(self.user_own_collection),
            'user_like_model': len(self.user_like_model),
            'user_like_dataset': len(self.user_like_dataset),
            'user_like_space': len(self.user_like_space),
            'user_follow_user': len(self.user_follow_user),
            'user_affiliatedWith_org': len(self.user_affiliatedWith_org),
            'user_follow_org': len(self.user_follow_org),
            'org_publish_model': len(self.org_publish_model),
            'org_publish_dataset': len(self.org_publish_dataset),
            'org_publish_space': len(self.org_publish_space),
            'org_own_collection': len(self.org_own_collection)
        }

        # 统计实体和关系的总数
        summary_stats = {
            'entities': sum(entity_stats.values()),
            'relations': sum(relation_stats.values())
        }

        # 记录实体统计信息
        logging.info("Entity stats:")
        for entity, count in entity_stats.items():
            logging.info(f"{entity}: {count}")
        
        # 记录关系统计信息
        logging.info("Relation stats:")
        for relation, count in relation_stats.items():
            logging.info(f"{relation}: {count}")
        
        # 记录总结统计信息
        logging.info("Summary stats:")
        for stat, count in summary_stats.items():
            logging.info(f"{stat}: {count}")
    
    def verify_relations(self):
        print("Verifying relations...")
        
        # 更新任务、模型、数据集、空间、论文、集合、用户和组织的 ID 集合
        self.task_ids = {task['id'] for task in self.processed_tasks}
        self.model_ids = {model['id'] for model in self.processed_models}
        self.dataset_ids = {dataset['id'] for dataset in self.processed_datasets}
        self.space_ids = {space['id'] for space in self.processed_spaces}
        self.paper_ids = {paper['id'] for paper in self.processed_papers}
        self.collection_slugs = {collection['slug'] for collection in self.processed_collections}
        self.user_ids = {user['id'] for user in self.processed_users}
        self.org_ids = {org['id'] for org in self.processed_orgs}
        
        # 定义关系检查列表，包含关系名称、关系数据、ID 键和有效 ID 集合
        relation_checks = [
            ('model_definedFor_task', self.model_definedFor_task, 'model_id', 'task_id', self.model_ids, self.task_ids),
            ('model_adapter_model', self.model_adapter_model, 'base_model_id', 'model_id', self.model_ids, self.model_ids),
            ('model_finetune_model', self.model_finetune_model, 'base_model_id', 'model_id', self.model_ids, self.model_ids),
            ('model_merge_model', self.model_merge_model, 'base_model_id', 'model_id', self.model_ids, self.model_ids),
            ('model_quantized_model', self.model_quantized_model, 'base_model_id', 'model_id', self.model_ids, self.model_ids),
            ('model_trainedOrFineTunedOn_dataset', self.model_trainedOrFineTunedOn_dataset, 'model_id', 'dataset_id', self.model_ids, self.dataset_ids),
            ('model_cite_paper', self.model_cite_paper, 'model_id', 'arxiv_id', self.model_ids, self.paper_ids),

            ('dataset_definedFor_task', self.dataset_definedFor_task, 'dataset_id', 'task_id', self.dataset_ids, self.task_ids),
            ('dataset_cite_paper', self.dataset_cite_paper, 'dataset_id', 'arxiv_id', self.dataset_ids, self.paper_ids),

            ('space_use_model', self.space_use_model, 'space_id', 'model_id', self.space_ids, self.model_ids),
            ('space_use_dataset', self.space_use_dataset, 'space_id', 'dataset_id', self.space_ids, self.dataset_ids),

            ('collection_contain_model', self.collection_contain_model, 'collection_slug', 'model_id', self.collection_slugs, self.model_ids),
            ('collection_contain_dataset', self.collection_contain_dataset, 'collection_slug', 'dataset_id', self.collection_slugs, self.dataset_ids),
            ('collection_contain_space', self.collection_contain_space, 'collection_slug', 'space_id', self.collection_slugs, self.space_ids),
            ('collection_contain_paper', self.collection_contain_paper, 'collection_slug', 'paper_id', self.collection_slugs, self.paper_ids),

            ('user_publish_model', self.user_publish_model, 'user_id', 'model_id', self.user_ids, self.model_ids),
            ('user_publish_dataset', self.user_publish_dataset, 'user_id', 'dataset_id', self.user_ids, self.dataset_ids),
            ('user_publish_space', self.user_publish_space, 'user_id', 'space_id', self.user_ids, self.space_ids),
            ('user_publish_paper', self.user_publish_paper, 'user_id', 'paper_id', self.user_ids, self.paper_ids),
            ('user_own_collection', self.user_own_collection, 'user_id', 'collection_slug', self.user_ids, self.collection_slugs),
            ('user_like_model', self.user_like_model, 'user_id', 'model_id', self.user_ids, self.model_ids),
            ('user_like_dataset', self.user_like_dataset, 'user_id', 'dataset_id', self.user_ids, self.dataset_ids),
            ('user_like_space', self.user_like_space, 'user_id', 'space_id', self.user_ids, self.space_ids),
            ('user_follow_user', self.user_follow_user, 'follower_id', 'followee_id', self.user_ids, self.user_ids),
            ('user_affiliatedWith_org', self.user_affiliatedWith_org, 'user_id', 'org_id', self.user_ids, self.org_ids),
            ('user_follow_org', self.user_follow_org, 'user_id', 'org_id', self.user_ids, self.org_ids),

            ('org_publish_model', self.org_publish_model, 'org_id', 'model_id', self.org_ids, self.model_ids),
            ('org_publish_dataset', self.org_publish_dataset, 'org_id', 'dataset_id', self.org_ids, self.dataset_ids),
            ('org_publish_space', self.org_publish_space, 'org_id', 'space_id', self.org_ids, self.space_ids),
            ('org_own_collection', self.org_own_collection, 'org_id', 'collection_slug', self.org_ids, self.collection_slugs)
        ]
    
        # 初始化无效关系计数
        invalid_relation_count = 0

        # 遍历关系检查列表，验证每个关系
        for relation_name, relations, id1_key, id2_key, valid_ids1, valid_ids2 in relation_checks:
            valid_relations = []
            for relation in relations:
                # 检查关系中的 ID 是否在有效 ID 集合中
                if relation[id1_key] in valid_ids1 and relation[id2_key] in valid_ids2:
                    valid_relations.append(relation)
                else:
                    # 记录无效关系的警告信息
                    logging.warning(f"[constructor] Invalid relation found in {relation_name}: {relation}")
                    invalid_relation_count += 1
            # 更新关系列表，只保留有效关系
            relations[:] = valid_relations

        if invalid_relation_count > 0:
            # 记录无效关系的总数
            logging.warning(f"[constructor] Found {invalid_relation_count} invalid relations")

        # save data after removing invalid relations
        # print(f"Saving data after removing invalid relations...")
        # self.save_relation_data()

    def process_task_data(self):
        try:
            print(f"Processing task data...")
            # 定义获取任务数据的 URL
            url = "https://huggingface.co/api/tasks"
            # 发送 GET 请求获取任务数据
            response = self.session.get(url)
            if response.status_code == 200:
                # 解析响应的 JSON 数据
                data = response.json()
                for task in data:
                    if data[task]['id'] not in self.task_ids:
                        # logging.info(f"[task] Found task: {data[task]['id']} - {data[task]['label']}")
                        # 添加任务 ID 到集合中
                        self.task_ids.add(data[task]['id'])
                        # 将任务信息添加到处理后的任务列表中
                        self.processed_tasks.append({
                            'id': data[task]['id'],
                            'label': data[task]['label']
                        })
            else:
                # 记录获取任务数据失败的错误信息
                logging.error(f"[task] Failed to fetch task details, status code: {response.status_code}")
            # 获取模型标签数据
            data = get_model_tags()
            if 'pipeline_tag' in data:
                for tag in data['pipeline_tag']:
                    if tag['id'] not in self.task_ids:
                        # logging.info(f"[task] Found task: {tag['id']} - {tag['label']}")
                        # 添加任务 ID 到集合中
                        self.task_ids.add(tag['id'])
                        # 将任务信息添加到处理后的任务列表中
                        self.processed_tasks.append({
                            'id': tag['id'],
                            'label': tag['label']
                        })
            # 获取数据集标签数据
            data = get_dataset_tags()
            if 'task_categories' in data:
                for tag in data['task_categories']:
                    tag_id = tag['id'].split(':')[1]
                    if tag_id not in self.task_ids:
                        # logging.info(f"[task] Found task: {tag_id} - {tag['label']}")
                        # 添加任务 ID 到集合中
                        self.task_ids.add(tag_id)
                        # 将任务信息添加到处理后的任务列表中
                        self.processed_tasks.append({
                            'id': tag_id,
                            'label': tag['label']
                        })
            
            print(f"Saving task data...")
            # 保存处理后的任务数据
            self.save_data('tasks', self.processed_tasks)
        except Exception as e:
            # 记录获取任务数据时发生异常的错误信息
            logging.error(f"[task] Exception occurred while fetching task details, error: {e}")
            return None

    def get_tag_classification(self, tag_type):
        if tag_type == 'model':
            # 获取模型标签数据
            data = get_model_tags()
        elif tag_type == 'dataset':
            # 获取数据集标签数据
            data = get_dataset_tags()
        else:
            # 抛出无效标签类型的异常
            raise ValueError(f"[{tag_type}] Invalid tag type for classification")

        # 初始化标签映射字典
        tag_mapping = {}
        for tag_type in data:
            for tag in data[tag_type]:
                # 构建标签映射，键为标签 ID，值为标签类型和标签名称
                tag_mapping[tag['id']] = {
                    'type': tag['type'],
                    'label': tag['label']
                }
        return tag_mapping

    def get_model_details(self, model, tag_mapping):
        """
        获取模型的详细信息，并构建模型数据字典，同时记录模型相关的各种关系。

        Args:
            model (object): 包含模型信息的对象，需包含 id、created_at、last_modified 等属性。
            tag_mapping (dict): 标签映射字典，用于将标签映射到对应的类型和标签名。

        Returns:
            dict: 包含模型详细信息的字典。
        """
        # logging.info(f"[model] Processing details for model: {model.id}")
        # 初始化模型数据字典，存储模型的基本信息
        model_data = {
            'id': model.id,
            'name': model.id.split('/')[1],
            'createdAt': model.created_at,
            'lastModified': model.last_modified,
            'downloads': model.downloads,
            'likes': model.likes,
            'region': None,
            'other': [],
            'libraries': [],
            'license': None,
            'languages': [],
            'pipeline_tag': [],
            'description': None
        }

        # 将模型作者添加到用户名 ID 集合中
        self.username_ids.add(model.author)
        # 记录用户名和其发布的模型 ID 的对应关系
        self.username_publish_model.append({
            'username': model.author,
            'model_id': model.id
        })
        # logging.info(f"[model] {model.id} is published by: {model.author}")

        # 检查模型是否有 pipeline 标签
        if model.pipeline_tag is not None:
            # 若标签存在于标签映射中，将映射后的标签名添加到模型数据中
            if model.pipeline_tag in tag_mapping:
                model_data['pipeline_tag'].append(tag_mapping[model.pipeline_tag]['label'])
            else:
                # 若标签不在映射中，直接添加原标签，并记录警告信息
                model_data['pipeline_tag'].append(model.pipeline_tag)
                logging.warning(f"[model] {model.id} Pipeline tag has no classification: {model.pipeline_tag}")
            # 记录模型和任务的对应关系
            self.model_definedFor_task.append({
                'model_id': model.id,
                'task_id': model.pipeline_tag
            })
        else:
            # 若模型没有 pipeline 标签，记录警告信息
            logging.warning(f"[model] Pipeline tag not found for model: {model.id}")

        # 检查模型是否有标签
        if model.tags is not None:
            # 遍历模型的所有标签
            for tag in model.tags:
                # 处理以 "dataset:" 开头的标签
                if tag.startswith("dataset:"):
                    # 提取数据集 ID
                    dataset_id = tag.split(':')[1]
                    # 记录模型和其训练或微调所用数据集的对应关系
                    self.model_trainedOrFineTunedOn_dataset.append({
                        'model_id': model.id,
                        'dataset_id': dataset_id
                    })
                    # logging.info(f"[model] {model.id} is trained/Fine-tuned on dataset: {dataset_id}")
                # 处理以 "arxiv:" 开头的标签
                elif tag.startswith("arxiv:"):
                    # 提取 Arxiv ID
                    arxiv_id = tag.split(':')[1]
                    # 将 Arxiv ID 添加到集合中
                    self.arxiv_ids.add(arxiv_id)
                    # 记录模型和其引用论文的对应关系
                    self.model_cite_paper.append({
                        'model_id': model.id,
                        'arxiv_id': arxiv_id
                    })
                    # logging.info(f"[model] {model.id} cite Arxiv: {arxiv_id}")
                # 处理以 "base_model:" 开头的标签
                elif tag.startswith("base_model:"):
                    # 分割标签字符串
                    parts = tag.split(":")
                    # 确保分割结果包含基础模型 ID 和关系类型
                    if len(parts) == 3:
                        base_model_id = parts[2]
                        relation = parts[1]
                        # 根据不同的关系类型记录模型间的关系
                        if relation == 'adapter':
                            self.model_adapter_model.append({
                                'base_model_id': base_model_id,
                                'model_id': model.id
                            })
                            # logging.info(f"[model] {model.id} is adapter model based on: {base_model_id}")
                        elif relation == 'finetune':
                            self.model_finetune_model.append({
                                'base_model_id': base_model_id,
                                'model_id': model.id
                            })
                            # logging.info(f"[model] {model.id} is fine-tuned model based on: {base_model_id}")
                        elif relation == 'merge':
                            self.model_merge_model.append({
                                'base_model_id': base_model_id,
                                'model_id': model.id
                            })
                            # logging.info(f"[model] {model.id} is merged model based on: {base_model_id}")
                        elif relation == 'quantized':
                            self.model_quantized_model.append({
                                'base_model_id': base_model_id,
                                'model_id': model.id
                            })
                            # logging.info(f"[model] {model.id} is quantized model based on: {base_model_id}")
                        else:
                            # 若遇到未知关系类型，记录错误信息（当前代码注释掉了）
                            # logging.error(f"[model] {model.id} has unknown relation: {relation} for base model: {base_model_id}")
                            pass
                else:
                    # 处理其他类型的标签
                    if tag in tag_mapping:
                        # 从标签映射中获取标签类型和标签名
                        tag_type = tag_mapping[tag]['type']
                        tag_label = tag_mapping[tag]['label']
                    else:
                        # 若标签不在映射中，将类型设为 'other'，标签名设为原标签
                        tag_type = 'other'
                        tag_label = tag
                    # 根据标签类型将标签信息添加到模型数据中
                    if tag_type == 'region':
                        model_data['region'] = tag_label
                    elif tag_type == 'other':
                        model_data['other'].append(tag_label)
                    elif tag_type == 'library':
                        model_data['libraries'].append(tag_label)
                    elif tag_type == 'license':
                        model_data['license'] = tag_label
                    elif tag_type == 'language':
                        model_data['languages'].append(tag_label)
                    elif tag_type == 'dataset':
                        # 数据集类型标签在此处不做处理
                        pass
                    elif tag_type == 'pipeline_tag':
                        # 若模型没有 pipeline 标签或当前标签和已有标签不同，添加标签信息并记录关系
                        if model.pipeline_tag is None or model.pipeline_tag != tag:
                            model_data['pipeline_tag'].append(tag_label)
                            self.model_definedFor_task.append({
                                'model_id': model.id,
                                'task_id': tag
                            })
                    else:
                        # 若遇到未知标签类型，记录错误信息
                        logging.error(f"[model] {model.id} has unknown tag type: {tag_type}")
        else:
            # 若模型没有标签，记录警告信息
            logging.warning(f"[model] Tags not found for model: {model.id}")

        # 返回包含模型详细信息的字典
        return model_data

    def save_model_data(self):
        """
        保存模型相关的数据到 JSON 文件。
        分别保存处理后的模型数据以及模型与其他实体之间的关系数据。
        """
        # 保存处理后的模型数据到名为 'models' 的 JSON 文件
        self.save_data('models', self.processed_models)
        # 保存模型与定义任务之间的关系数据到 'model_definedFor_task' 的 JSON 文件
        self.save_data('model_definedFor_task', self.model_definedFor_task)
        # 保存模型与适配器模型之间的关系数据到 'model_adapter_model' 的 JSON 文件
        self.save_data('model_adapter_model', self.model_adapter_model)
        # 保存模型与微调模型之间的关系数据到 'model_finetune_model' 的 JSON 文件
        self.save_data('model_finetune_model', self.model_finetune_model)
        # 保存模型与合并模型之间的关系数据到 'model_merge_model' 的 JSON 文件
        self.save_data('model_merge_model', self.model_merge_model)
        # 保存模型与量化模型之间的关系数据到 'model_quantized_model' 的 JSON 文件
        self.save_data('model_quantized_model', self.model_quantized_model)
        # 保存模型与训练或微调所使用的数据集之间的关系数据到 'model_trainedOrFineTunedOn_dataset' 的 JSON 文件
        self.save_data('model_trainedOrFineTunedOn_dataset', self.model_trainedOrFineTunedOn_dataset)
        # 保存模型与引用论文之间的关系数据到 'model_cite_paper' 的 JSON 文件
        self.save_data('model_cite_paper', self.model_cite_paper)

    def process_model_data(self):
        """
        处理模型数据，获取模型标签分类，遍历模型获取详细信息，保存模型数据并记录模型 ID。
        """
        # 打印提示信息，表明正在准备模型数据
        print(f"Preparing model data...")
        # 获取模型标签的分类映射
        tag_mapping = self.get_tag_classification('model')
        # 获取最多 1000 个完整的模型信息
        models = list_models(full=True, limit=1000)
        
        # 打印提示信息，表明正在处理模型数据
        print(f"Processing models...")
        # 使用 tqdm 显示进度条，遍历所有模型
        for model in tqdm(models):
            # 获取单个模型的详细信息
            model_data = self.get_model_details(model, tag_mapping)
            # 将处理后的模型数据添加到已处理模型列表中
            self.processed_models.append(model_data)
        
        # 打印提示信息，表明正在保存模型数据
        print(f"Saving model data...")
        # 调用保存模型数据的方法
        self.save_model_data()
        # 提取已处理模型的 ID，存储在集合中
        self.model_ids = {model['id'] for model in self.processed_models}

    def get_model_description(self, model):
        """
        获取单个模型的 README.md 文件内容作为模型描述。
        如果文件过大或请求失败，会记录相应的日志信息。

        Args:
            model (dict): 包含模型信息的字典，需包含 'id' 字段。
        """
        try:
            # 打印日志信息，表明正在获取指定模型的描述信息（当前代码行被注释）
            # logging.info(f"[model] Fetching description for model: {model['id']}")
            # 构建模型 README.md 文件的 URL
            url = f"https://huggingface.co/{model['id']}/resolve/main/README.md"
            # 发送 HEAD 请求，获取文件的基本信息，允许重定向
            head_response = self.session.head(url, allow_redirects=True)
            # 如果 HEAD 请求状态码为 200，表示文件存在
            if head_response.status_code == 200:
                # 获取文件的内容长度，若不存在则默认为 0
                content_length = int(head_response.headers.get('Content-Length', 0))
                # 如果文件大小超过 100MB
                if content_length > 100 * 1024 * 1024:  # 100MB
                    # 记录警告日志，表明文件过大，跳过获取操作
                    logging.warning(f"[model] Description for model {model['id']} is too large ({content_length} bytes), skipping.")
                    return
                # 发送 GET 请求，获取文件内容
                response = self.session.get(url)
                # 如果 GET 请求状态码为 200，表示成功获取文件内容
                if response.status_code == 200:
                    # 将文件内容添加到模型信息中
                    model['description'] = response.text
            # 如果 HEAD 请求状态码为 404，表示文件不存在
            elif head_response.status_code == 404:
                # 记录警告日志，表明未找到模型的描述文件
                logging.warning(f"[model] Description not found for model: {model['id']}")
            else:
                # 记录错误日志，表明获取模型描述文件失败，并记录状态码
                logging.error(f"[model] Failed to fetch description for model: {model['id']}, status code: {head_response.status_code}")
        except Exception as e:
            # 记录错误日志，表明获取模型描述时发生异常，并记录异常信息
            logging.error(f"[model] Exception occurred while fetching description for model: {model['id']}, error: {e}")

    def process_model_description(self):
        """
        多线程批量获取已处理模型的描述信息，并保存更新后的模型数据。
        """
        # 打印提示信息，表明正在获取模型描述信息
        print(f"Fetching model descriptions...")
        # 使用线程池执行并发任务
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 使用 tqdm 显示进度条，调用 get_model_description 方法处理每个已处理的模型
            list(tqdm(
                executor.map(self.get_model_description, self.processed_models),
                total=len(self.processed_models)
            ))
        
        # 打印提示信息，表明正在保存模型数据
        print(f"Saving model data...")
        # 将更新后的已处理模型数据保存到 'models' 的 JSON 文件中
        self.save_data('models', self.processed_models)

    def get_dataset_details(self, dataset, tag_mapping):
        """
        获取数据集的详细信息，并构建数据集数据字典，同时记录数据集与其他实体之间的关系。

        Args:
            dataset (object): 包含数据集信息的对象，需包含 id、created_at、last_modified、author、tags 等属性。
            tag_mapping (dict): 标签映射字典，用于将标签映射到对应的类型和标签名。

        Returns:
            dict: 包含数据集详细信息的字典。
        """
        # logging.info(f"[dataset] Processing details for dataset: {dataset.id}")
        # 初始化数据集数据字典，存储数据集的基本信息和各种属性
        dataset_data = {
            'id': dataset.id,
            'name': dataset.id.split('/')[1],
            'createdAt': dataset.created_at,
            'lastModified': dataset.last_modified,
            'downloads': dataset.downloads,
            'likes': dataset.likes,
            'libraries': [],
            'license': None,
            'languages': [],
            'other': [],
            'sub-tasks': [],
            'tasks': [],
            'size': None,
            'formats': [],
            'modalities': [],
            'tags': [],
            'description': None
        }

        # 将数据集作者添加到用户名 ID 集合中
        self.username_ids.add(dataset.author)
        # 记录用户名和其发布的数据集 ID 的关系
        self.username_publish_dataset.append({
            'username': dataset.author,
            'dataset_id': dataset.id
        })
        # logging.info(f"[dataset] {dataset.id} is published by: {dataset.author}")

        # 检查数据集是否有标签
        if dataset.tags is not None:
            # 遍历数据集中的所有标签
            for tag in dataset.tags:
                # 检查标签是否以 "arxiv:" 开头
                if tag.startswith("arxiv:"):
                    # 提取 Arxiv ID
                    arxiv_id = tag.split(':')[1]
                    # 将 Arxiv ID 添加到集合中
                    self.arxiv_ids.add(arxiv_id)
                    # 记录数据集引用论文的关系
                    self.dataset_cite_paper.append({
                        'dataset_id': dataset.id,
                        'arxiv_id': arxiv_id
                    })
                    # logging.info(f"[dataset] {dataset.id} cite Arxiv: {arxiv_id}")
                else:
                    # 检查标签是否以 "task_categories:" 开头
                    if tag.startswith("task_categories:"):
                        # 提取任务 ID
                        task_id = tag.split(':')[1]
                        # 记录数据集与任务的关系
                        self.dataset_definedFor_task.append({
                            'dataset_id': dataset.id,
                            'task_id': task_id
                        })
                    # 检查标签是否在标签映射字典中
                    if tag in tag_mapping:
                        # 获取标签类型
                        tag_type = tag_mapping[tag]['type']
                        # 获取标签名
                        tag_label = tag_mapping[tag]['label']
                    else:
                        # 若标签不在映射字典中，默认标签类型为 'tag'
                        tag_type = 'tag'
                        # 默认标签名为标签本身
                        tag_label = tag
                    # 根据标签类型将标签添加到数据集数据字典的对应字段中
                    if tag_type == 'library':
                        dataset_data['libraries'].append(tag_label)
                    elif tag_type == 'license':
                        dataset_data['license'] = tag_label
                    elif tag_type == 'language':
                        dataset_data['languages'].append(tag_label)
                    elif tag_type == 'other':
                        dataset_data['other'].append(tag_label)
                    elif tag_type == 'task_ids':
                        dataset_data['sub-tasks'].append(tag_label)
                    elif tag_type == 'task_categories':
                        dataset_data['tasks'].append(tag_label)
                    elif tag_type == 'size_categories':
                        dataset_data['size'] = tag_label
                    elif tag_type == 'format':
                        dataset_data['formats'].append(tag_label)
                    elif tag_type == 'modality':
                        dataset_data['modalities'].append(tag_label)
                    elif tag_type == 'tag':
                        dataset_data['tags'].append(tag_label)
                    else:
                        # 若遇到未知标签类型，记录错误日志
                        logging.error(f"[dataset] {dataset.id} has unknown tag type: {tag_type}")
        else:
            # 若数据集没有标签，记录警告日志
            logging.warning(f"[dataset] Tags not found for dataset: {dataset.id}")

        return dataset_data

    def save_dataset_data(self):
        """
        保存数据集相关的数据到 JSON 文件。
        分别保存处理后的数据集数据、数据集与任务的关系数据以及数据集引用论文的关系数据。
        """
        # 保存处理后的数据集数据
        self.save_data('datasets', self.processed_datasets)
        # 保存数据集与定义任务的关系数据
        self.save_data('dataset_definedFor_task', self.dataset_definedFor_task)
        # 保存数据集引用论文的关系数据
        self.save_data('dataset_cite_paper', self.dataset_cite_paper)

    def process_dataset_data(self):
        """
        处理数据集数据，获取数据集标签分类，遍历数据集获取详细信息，
        保存数据集相关数据，并记录数据集 ID。
        """
        print(f"Preparing dataset data...")
        # 获取数据集标签分类映射
        tag_mapping = self.get_tag_classification('dataset')
        # 获取最多 1000 个完整的数据集信息
        datasets = list_datasets(full=True, limit=1000)

        print(f"Processing datasets...")
        # 遍历所有数据集，显示进度条
        for dataset in tqdm(datasets):
            # 获取单个数据集的详细信息
            dataset_data = self.get_dataset_details(dataset, tag_mapping)
            # 将处理后的数据集信息添加到已处理列表中
            self.processed_datasets.append(dataset_data)
        
        # 保存数据集相关数据
        print(f"Saving dataset data...")
        self.save_dataset_data()
        # 获取所有已处理数据集的 ID
        self.dataset_ids = {dataset['id'] for dataset in self.processed_datasets}

    def get_dataset_description(self, dataset):
        """
        获取单个数据集的 README.md 文件内容作为数据集描述。
        如果文件过大或请求失败，会记录相应的日志信息。

        Args:
            dataset (dict): 包含数据集信息的字典，需包含 'id' 字段。
        """
        try:
            # logging.info(f"[dataset] Fetching description for dataset: {dataset['id']}")
            # 构建获取数据集 README.md 文件的 URL
            url = f"https://huggingface.co/datasets/{dataset['id']}/resolve/main/README.md"
            # 发送 HEAD 请求，获取文件元信息
            head_response = self.session.head(url, allow_redirects=True)
            if head_response.status_code == 200:
                # 获取文件大小
                content_length = int(head_response.headers.get('Content-Length', 0))
                if content_length > 100 * 1024 * 1024:  # 100MB
                    # 文件过大，记录警告日志并跳过
                    logging.warning(f"[dataset] Description for dataset {dataset['id']} is too large ({content_length} bytes), skipping.")
                    return
                # 发送 GET 请求，获取文件内容
                response = self.session.get(url)
                if response.status_code == 200:
                    # 将获取到的描述信息添加到数据集信息中
                    dataset['description'] = response.text
            elif head_response.status_code == 404:
                # 文件未找到，记录警告日志
                logging.warning(f"[dataset] Description not found for dataset: {dataset['id']}")
            else:
                # 请求失败，记录错误日志
                logging.error(f"[dataset] Failed to fetch description for dataset: {dataset['id']}, status code: {head_response.status_code}")
        except Exception as e:
            # 请求过程中出现异常，记录错误日志
            logging.error(f"[dataset] Exception occurred while fetching description for dataset: {dataset['id']}, error: {e}")

    def process_dataset_description(self):
        """
        多线程批量获取已处理数据集的描述信息，并保存更新后的数据集数据。
        """
        print(f"Fetching dataset descriptions...")
        # 使用线程池多线程获取数据集描述信息
        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(tqdm(
                executor.map(self.get_dataset_description, self.processed_datasets),
                total=len(self.processed_datasets)
            ))
        
        # 保存更新后的数据集数据
        print(f"Saving dataset data...")
        self.save_data('datasets', self.processed_datasets)

    def get_space_details(self, space_id):
        """
        获取单个空间的详细信息，并记录空间与发布者、模型、数据集之间的关系。

        Args:
            space_id (str): 空间的 ID。

        Returns:
            dict: 包含空间详细信息的字典，如果请求失败则返回 None。
        """
        try:
            # logging.info(f"[space] Processing details for space: {space_id}")
            # 构建获取空间信息的 API URL
            url = f"https://huggingface.co/api/spaces/{space_id}"
            # 发送 GET 请求，获取空间信息
            response = self.session.get(url)
            if response.status_code == 200:
                # 将响应内容解析为 JSON 格式
                data = response.json()
                # 构建空间信息字典
                space_data = {
                    'id': data['id'],
                    'name': data['id'].split('/')[1],
                    'createdAt': data['created_at'] if 'created_at' in data else None,
                    'lastModified': data['last_modified'] if 'last_modified' in data else None,
                    'likes': data['likes'] if 'likes' in data else 0,
                    'tags': []
                }

                # 获取空间发布者信息
                space_publisher = data['author']
                # 将发布者 ID 添加到用户名 ID 集合中
                self.username_ids.add(space_publisher)
                # 记录用户名与发布空间的关系
                self.username_publish_space.append({
                    'username': space_publisher,
                    'space_id': data['id']
                })
                # logging.info(f"[space] {space_id} is published by: {space_publisher}")

                if 'tags' in data:
                    # 将空间标签添加到空间信息字典中
                    for tag in data['tags']:
                        space_data['tags'].append(tag)
                else:
                    # 未找到标签，记录警告日志
                    logging.warning(f"[space] Tags not found for space: {space_id}")
                if 'models' in data:
                    # 记录空间使用模型的关系
                    for model_id in data['models']:
                        self.space_use_model.append({
                            'space_id': data['id'],
                            'model_id': model_id
                        })
                        # logging.info(f"[space] {space_id} uses model: {model_id}")
                if 'datasets' in data:
                    # 记录空间使用数据集的关系
                    for dataset_id in data['datasets']:
                        self.space_use_dataset.append({
                            'space_id': data['id'],
                            'dataset_id': dataset_id
                        })
                        # logging.info(f"[space] {space_id} uses dataset: {dataset_id}")
                return space_data
            else:
                # 请求失败，记录错误日志
                logging.error(f"[space] Failed to fetch space details for {space_id}, status code: {response.status_code}")
                return None
        except Exception as e:
            # 请求过程中出现异常，记录错误日志
            logging.error(f"[space] Exception occurred while fetching space details for {space_id}, error: {e}")
            return None

    def save_space_data(self):
        """
        保存空间相关数据到 JSON 文件。
        分别保存处理后的空间数据、空间使用模型的关系数据和空间使用数据集的关系数据。
        """
        # 保存处理后的空间数据到名为 'spaces' 的 JSON 文件
        self.save_data('spaces', self.processed_spaces)
        # 保存空间使用模型的关系数据到名为 'space_use_model' 的 JSON 文件
        self.save_data('space_use_model', self.space_use_model)
        # 保存空间使用数据集的关系数据到名为 'space_use_dataset' 的 JSON 文件
        self.save_data('space_use_dataset', self.space_use_dataset)

    def process_space_data(self):
        """
        处理 Hugging Face 上的空间数据。
        获取空间列表，多线程处理每个空间的详细信息，保存处理结果并记录有效空间 ID。
        """
        print(f"Preparing space data...")
        # 获取最多 1000 个空间的列表
        spaces = list_spaces(limit=1000)
        # 从空间列表中提取所有空间的 ID
        space_ids = [space.id for space in tqdm(spaces)]
        
        print(f"Processing spaces...")
        # 使用线程池多线程处理每个空间的详细信息
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(self.get_space_details, space_ids),
                total=len(space_ids)
            ))

        # 过滤掉处理失败（返回 None）的结果，得到有效的处理后空间数据
        self.processed_spaces = [space for space in results if space is not None]
        
        # 保存空间相关数据
        print(f"Saving space data...")
        self.save_space_data()
        # 提取有效空间数据中的 ID 并存储为集合
        self.space_ids = {space['id'] for space in self.processed_spaces}

    def get_collection_details(self, collection_slug):
        """
        获取单个集合的详细信息，并记录集合与所有者、模型、数据集、空间和论文之间的关系。

        Args:
            collection_slug (str): 集合的标识符。

        Returns:
            dict: 包含集合详细信息的字典，如果请求失败则返回 None。
        """
        try:
            # logging.info(f"[collection] Processing details for collection: {collection_slug}")
            # 构建获取集合详细信息的 API 请求 URL
            url = f"https://huggingface.co/api/collections/{collection_slug}"
            # 发送 HTTP 请求获取集合详细信息，不携带授权头
            response = self.session.get(url, headers={'Authorization': None})
            if response.status_code == 200:
                # 将响应内容解析为 JSON 格式
                data = response.json()
                # 构建集合数据字典，初始化基本信息
                collection_data = {
                    'slug': data['slug'],
                    'title': data['title'],
                    'upvotes': data['upvotes'] if 'upvotes' in data else 0,
                    'description': None
                }

                # 获取集合所有者的名称
                collection_owner = data['owner']['name']
                # 获取集合所有者的类型
                collection_owner_type = data['owner']['type']
                # 将集合所有者名称添加到用户名 ID 集合中
                self.username_ids.add(collection_owner)

                if collection_owner_type == 'user':
                    # 如果所有者类型是用户，将其添加到用户 ID 集合中
                    self.user_ids.add(collection_owner)
                    # 记录用户拥有集合的关系
                    self.user_own_collection.append({
                        'user_id': collection_owner,
                        'collection_slug': data['slug']
                    })
                    # logging.info(f"[collection] {collection_slug} is owned by user: {collection_owner}")
                elif collection_owner_type == 'org':
                    # 如果所有者类型是组织，将其添加到组织 ID 集合中
                    self.org_ids.add(collection_owner)
                    # 记录组织拥有集合的关系
                    self.org_own_collection.append({
                        'org_id': collection_owner,
                        'collection_slug': data['slug']
                    })
                    # logging.info(f"[collection] {collection_slug} is owned by org: {collection_owner}")
                else:
                    # 如果所有者类型未知，记录错误日志
                    logging.error(f"[collection] Unknown owner type: {collection_owner_type} for collection: {collection_slug}")

                if 'description' in data:
                    # 如果响应中包含描述信息，将其添加到集合数据字典中
                    collection_data['description'] = data['description']
                else:
                    # 如果响应中不包含描述信息，记录警告日志
                    logging.warning(f"[collection] Description not found for collection: {collection_slug}")
                
                if 'items' in data:
                    # 如果响应中包含集合中的项目信息，遍历每个项目
                    for item in data['items']:
                        if item['type'] == 'model':
                            # 如果项目类型是模型，记录集合包含模型的关系
                            self.collection_contain_model.append({
                                'collection_slug': data['slug'],
                                'model_id': item['id']
                            })
                            # logging.info(f"[collection] {collection_slug} contains model: {item['id']}")
                        elif item['type'] == 'dataset':
                            # 如果项目类型是数据集，记录集合包含数据集的关系
                            self.collection_contain_dataset.append({
                                'collection_slug': data['slug'],
                                'dataset_id': item['id']
                            })
                            # logging.info(f"[collection] {collection_slug} contains dataset: {item['id']}")
                        elif item['type'] == 'space':
                            # 如果项目类型是空间，记录集合包含空间的关系
                            self.collection_contain_space.append({
                                'collection_slug': data['slug'],
                                'space_id': item['id']
                            })
                            # logging.info(f"[collection] {collection_slug} contains space: {item['id']}")
                        elif item['type'] == 'paper':
                            # 如果项目类型是论文，记录集合包含论文的关系
                            self.collection_contain_paper.append({
                                'collection_slug': data['slug'],
                                'paper_id': item['id']
                            })
                            # 将论文的 arXiv ID 添加到集合中
                            self.arxiv_ids.add(item['id'])
                            # logging.info(f"[collection] {collection_slug} contains paper: {item['id']}")
                        else:
                            # 如果项目类型未知，记录错误日志
                            logging.error(f"[collection] Unknown item type: {item['type']} for collection: {collection_slug}")
                else:
                    # 如果响应中不包含项目信息，记录警告日志
                    logging.warning(f"[collection] Items not found for collection: {collection_slug}")
                return collection_data
            else:
                # 如果请求状态码不是 200，记录错误日志
                logging.error(f"[collection] Failed to fetch collection details for {collection_slug}, status code: {response.status_code}")
                return None
        except Exception as e:
            # 如果请求过程中发生异常，记录错误日志
            logging.error(f"[collection] Exception occurred while fetching collection details for {collection_slug}, error: {e}")
            return None

    def save_collection_data(self):
        """
        保存集合相关的数据到 JSON 文件。
        分别保存处理后的集合数据、集合包含的模型、数据集、空间、论文关系数据，
        以及用户和组织拥有集合的关系数据。
        """
        self.save_data('collections', self.processed_collections)
        self.save_data('collection_contain_model', self.collection_contain_model)
        self.save_data('collection_contain_dataset', self.collection_contain_dataset)
        self.save_data('collection_contain_space', self.collection_contain_space)
        self.save_data('collection_contain_paper', self.collection_contain_paper)
        self.save_data('user_own_collection', self.user_own_collection)
        self.save_data('org_own_collection', self.org_own_collection)

    def get_collection_slugs(self, limit=None):
        """
        获取 Hugging Face 上集合的标识符列表。

        Args:
            limit (int, optional): 获取集合标识符的数量上限。默认为 None，表示不限制数量。

        Returns:
            list: 集合标识符列表，如果出现错误则返回空列表。
        """
        try:
            # logging.info(f"[collection] Fetching collection slugs...")
            collection_slugs = []
            url = "https://huggingface.co/api/collections"
            # 发起请求获取集合信息，设置每页返回 100 条数据，不携带授权信息
            response = self.session.get(url, params={'limit': 100}, headers={'Authorization': None})
            if response.status_code != 200:
                logging.error(f"[collection] Failed to fetch collection slugs from {url}, status code: {response.status_code}")
                return []
            # 从响应中提取集合标识符并添加到列表中
            collection_slugs.extend([collection.get('slug') for collection in response.json()])
            # 获取下一页的 URL
            next_page = response.links.get('next', {}).get('url')
            # 循环获取下一页数据，直到没有下一页或达到数量限制
            while next_page is not None and (limit is None or len(collection_slugs) < limit):
                response = self.session.get(next_page, headers={'Authorization': None})
                if response.status_code != 200:
                    logging.error(f"[collection] Failed to fetch collection slugs from {url}, so far fetched: {len(collection_slugs)}, status code: {response.status_code}")
                    return collection_slugs
                collection_slugs.extend([collection.get('slug') for collection in response.json()])
                next_page = response.links.get('next', {}).get('url')
            # 根据是否有数量限制返回对应的数据
            return collection_slugs if limit is None else collection_slugs[:limit]
        except Exception as e:
            logging.error(f"[collection] Exception occurred while fetching collection slugs, error: {e}")
            return []

    def process_collection_data(self):
        """
        处理 Hugging Face 上的集合数据。
        获取集合标识符列表，多线程处理每个集合的详细信息，
        保存处理结果并记录有效集合的标识符。
        """
        print(f"Preparing collection data...")
        # 获取集合标识符列表，最多获取 100 个
        collection_slugs = self.get_collection_slugs(limit=100)

        print(f"Processing collections...")
        # 使用线程池多线程处理每个集合的详细信息
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(self.get_collection_details, collection_slugs),
                total=len(collection_slugs)
            ))
        
        # 过滤掉获取失败的集合数据
        self.processed_collections = [collection for collection in results if collection is not None]
        
        # Save collection related data
        print(f"Saving collection data...")
        # 保存集合相关数据
        self.save_collection_data()
        # 记录有效集合的标识符
        self.collection_slugs = {collection['slug'] for collection in self.processed_collections}

    def get_paper_details(self, arxiv_id):
        """
        获取单个论文的详细信息，并记录作者与论文的发布关系。

        Args:
            arxiv_id (str): 论文的 arXiv ID。

        Returns:
            dict: 包含论文详细信息的字典，如果请求失败则返回 None。
        """
        try:
            # logging.info(f"[paper] Processing details for paper: {arxiv_id}")
            url = f"https://huggingface.co/api/papers/{arxiv_id}"
            response = self.session.get(url)
            if response.status_code == 200:
                data = response.json()
                paper_data = {
                    'id': data['id'],
                    'authors': [author['name'] for author in data['authors']] if 'authors' in data else [],
                    'publishedAt': data['publishedAt'] if 'publishedAt' in data else None,
                    'title': data['title'] if 'title' in data else None,
                    'summary': data['summary'] if 'summary' in data else None,
                    'upvotes': data['upvotes'] if 'upvotes' in data else 0
                }
                if 'authors' in data:
                    for author in data['authors']:
                        if 'user' in author:
                            user_id = author['user']['user']
                            # 记录用户 ID
                            self.user_ids.add(user_id)
                            # 记录用户发布论文的关系
                            self.user_publish_paper.append({
                                'user_id': user_id,
                                'paper_id': data['id']
                            })
                            # logging.info(f"[paper] {arxiv_id} has a registered author: {user_id}")
                return paper_data
            else:
                logging.error(f"[paper] Failed to fetch paper details for {arxiv_id}, status code: {response.status_code}")
                return None
        except Exception as e:
            logging.error(f"[paper] Exception occurred while fetching paper details for {arxiv_id}, error: {e}")
            return None

    def save_paper_data(self):
        """
        保存论文相关的数据到 JSON 文件。
        分别保存处理后的论文数据和用户发布论文的关系数据。
        """
        self.save_data('papers', self.processed_papers)
        self.save_data('user_publish_paper', self.user_publish_paper)

    def process_paper_data(self):
        """
        处理论文数据。
        多线程获取每个论文的详细信息，保存处理结果并记录有效论文的 ID。
        """
        print(f"Processing papers...")
        # 使用线程池多线程处理每个论文的详细信息
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(self.get_paper_details, self.arxiv_ids),
                total=len(self.arxiv_ids)
            ))
        
        # 过滤掉获取失败的论文数据
        self.processed_papers = [paper for paper in results if paper is not None]

        # Save paper related data
        print(f"Saving paper data...")
        # 保存论文相关数据
        self.save_paper_data()
        # 记录有效论文的 ID
        self.paper_ids = {paper['id'] for paper in self.processed_papers}
    
    def get_user_like_by_repo(self, repo_id, repo_type):
        """
        获取指定仓库的点赞用户信息，并记录用户与仓库的点赞关系。

        Args:
            repo_id (str): 仓库的 ID。
            repo_type (str): 仓库的类型，如 'model', 'dataset' 等。

        Returns:
            list: 包含用户点赞信息的列表，如果请求失败则返回空列表。
        """
        try:
            # logging.info(f"[user_like] Processing like for {repo_type}: {repo_id}")
            user_like_repo = []
            url = f"https://huggingface.co/api/{repo_type}s/{repo_id}/likers"
            response = self.session.get(url)
            if response.status_code != 200:
                logging.error(f"[user_like] Failed to fetch likes for {repo_type}: {repo_id}, status code: {response.status_code}")
                return []
            for user in response.json():
                # 记录用户 ID
                self.user_ids.add(user['user'])
                # 记录用户点赞仓库的关系
                user_like_repo.append({
                    'user_id': user['user'],
                    f'{repo_type}_id': repo_id
                })
                # logging.info(f"[user_like] {user['user']} liked {repo_type}: {repo_id}")
            # 获取下一页的 URL
            next_page = response.links.get('next', {}).get('url')
            while next_page is not None:
                response = self.session.get(next_page)
                if response.status_code != 200:
                    logging.error(f"[user_like] Failed to fetch likes for {repo_type}: {repo_id} from {url}, so far fetched: {len(user_like_repo)}, status code: {response.status_code}")
                    return user_like_repo
                for user in response.json():
                    self.user_ids.add(user['user'])
                    user_like_repo.append({
                        'user_id': user['user'],
                        f'{repo_type}_id': repo_id
                    })
                    # logging.info(f"[user_like] {user['user']} liked {repo_type}: {repo_id}")
                next_page = response.links.get('next', {}).get('url')
            return user_like_repo
        except Exception as e:
            logging.error(f"[user_like] Exception occurred while fetching likes for {repo_type}: {repo_id}, error: {e}")
            return []

    def save_user_like_data(self):
        """
        保存用户对不同类型仓库的点赞数据到 JSON 文件。
        分别保存用户对模型、数据集和空间的点赞数据。
        """
        self.save_data('user_like_model', self.user_like_model)  # 保存用户对模型的点赞数据
        self.save_data('user_like_dataset', self.user_like_dataset)  # 保存用户对数据集的点赞数据
        self.save_data('user_like_space', self.user_like_space)  # 保存用户对空间的点赞数据

    def process_user_like_repo(self):
        """
        处理用户对不同类型仓库的点赞数据。
        多线程获取每种仓库类型下各个仓库的用户点赞信息，最后保存点赞数据。
        """
        for repo_type, repo_ids, result_list in [
            ('model', self.model_ids, self.user_like_model),
            ('dataset', self.dataset_ids, self.user_like_dataset),
            ('space', self.space_ids, self.user_like_space)
        ]:
            print(f"Processing likes for {repo_type}s...")  # 打印正在处理的仓库类型
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # 多线程调用 get_user_like_by_repo 方法获取每个仓库的点赞信息
                results = list(tqdm(
                    executor.map(lambda repo_id: self.get_user_like_by_repo(repo_id, repo_type), repo_ids),
                    total=len(repo_ids)
                ))
            
            for result in results:
                result_list.extend(result)  # 将获取到的点赞信息添加到对应的结果列表中
        
        # Save user like related data
        print(f"Saving user like data...")  # 打印正在保存用户点赞数据的信息
        self.save_user_like_data()  # 调用保存点赞数据的方法

    def username_classification(self, username):
        """
        对用户名进行分类，判断其是用户还是组织。
        向 Hugging Face API 发送请求，根据响应状态码将用户名添加到对应的集合中。

        Args:
            username (str): 需要分类的用户名或组织名。
        """
        try:
            # logging.info(f"[user&org] Classifying user/organization: {username}")
            if username in self.user_ids or username in self.org_ids:
                return  # 如果用户名已经在用户或组织集合中，直接返回
            url = f"https://huggingface.co/api/users/{username}/overview"  # 构建获取用户信息的 API URL
            response = self.session.get(url)  # 发送 GET 请求
            if response.status_code not in [200, 404]:
                # 若状态码不是 200 或 404，记录错误信息
                logging.error(f"[user&org] Failed to classify user/organization: {username} as user, status code: {response.status_code}")
                return
            if response.status_code == 200:
                self.user_ids.add(username)  # 若状态码为 200，将用户名添加到用户集合中
            url = f"https://huggingface.co/api/organizations/{username}/overview"  # 构建获取组织信息的 API URL
            response = self.session.get(url)  # 发送 GET 请求
            if response.status_code not in [200, 404]:
                # 若状态码不是 200 或 404，记录错误信息
                logging.error(f"[user&org] Failed to classify user/organization: {username} as organization, status code: {response.status_code}")
                return
            if response.status_code == 200:
                self.org_ids.add(username)  # 若状态码为 200，将用户名添加到组织集合中
            if username not in self.user_ids and username not in self.org_ids:
                # 若用户名既不在用户集合也不在组织集合中，记录错误信息
                logging.error(f"[user&org] Failed to classify user/organization: {username} as user/organization")
        except Exception as e:
            # 捕获异常并记录错误信息
            logging.error(f"[user&org] Exception occurred while fetching user details for {username}, error: {e}")
        
    def get_user_overview(self, user_id):
        """
        获取单个用户的概览信息。
        向 Hugging Face API 发送请求，若请求成功则返回用户信息，否则记录错误信息并返回 None。

        Args:
            user_id (str): 用户的 ID。

        Returns:
            dict: 包含用户 ID 和全名的字典，如果请求失败则返回 None。
        """
        try:
            # logging.info(f"[user] Processing details for user: {user_id}")
            url = f"https://huggingface.co/api/users/{user_id}/overview"  # 构建获取用户概览信息的 API URL
            response = self.session.get(url)  # 发送 GET 请求
            if response.status_code == 200:
                data = response.json()  # 将响应内容解析为 JSON
                user_data = {
                    'id': data['user'],
                    'fullname': data['fullname'] if 'fullname' in data else None  # 若存在全名则获取，否则为 None
                }
                return user_data  # 返回用户信息
            else:
                # 若状态码不为 200，记录错误信息
                logging.error(f"[user] Failed to fetch user details for {user_id}, status code: {response.status_code}")
                return None
        except Exception as e:
            # 捕获异常并记录错误信息
            logging.error(f"[user] Exception occurred while fetching user details for {user_id}, error: {e}")
            return None

    def get_org_overview(self, org_id):
        """
        获取单个组织的概览信息。
        向 Hugging Face API 发送请求，若请求成功则返回组织信息，否则记录错误信息并返回 None。

        Args:
            org_id (str): 组织的 ID。

        Returns:
            dict: 包含组织 ID 和全名的字典，如果请求失败则返回 None。
        """
        try:
            # logging.info(f"[org] Processing details for org: {org_id}")
            url = f"https://huggingface.co/api/organizations/{org_id}/overview"  # 构建获取组织概览信息的 API URL
            response = self.session.get(url)  # 发送 GET 请求
            if response.status_code == 200:
                data = response.json()  # 将响应内容解析为 JSON
                org_data = {
                    'id': data['name'],
                    'fullname': data['fullname'] if 'fullname' in data else None  # 若存在全名则获取，否则为 None
                }
                return org_data  # 返回组织信息
            else:
                # 若状态码不为 200，记录错误信息
                logging.error(f"[org] Failed to fetch org details for {org_id}, status code: {response.status_code}")
                return None
        except Exception as e:
            # 捕获异常并记录错误信息
            logging.error(f"[org] Exception occurred while fetching org details for {org_id}, error: {e}")
            return None
        
    def get_user_followers(self, user_id):
        """
        获取指定用户的所有关注者信息。支持分页获取，若请求失败会记录错误日志。

        Args:
            user_id (str): 用户的 ID。

        Returns:
            list: 包含关注者信息的列表，每个元素是一个字典，包含关注者 ID 和被关注者 ID；
                  若请求失败则返回空列表。
        """
        try:
            # logging.info(f"[user] Processing followers for user: {user_id}")
            # 用于存储用户的关注者信息
            user_followers = []
            # 构建获取用户关注者信息的 API 请求 URL
            url = f"https://huggingface.co/api/users/{user_id}/followers"
            # 发送请求获取关注者信息
            response = self.session.get(url)
            # 检查响应状态码，若不是 200 表示请求失败
            if response.status_code != 200:
                logging.error(f"[user] Failed to fetch followers for user: {user_id}, status code: {response.status_code}")
                return []
            # 遍历响应数据，提取关注者信息
            for follower in response.json():
                # self.user_ids.add(follower['user'])
                user_followers.append({
                    'follower_id': follower['user'],  # 关注者的 ID
                    'followee_id': user_id  # 被关注者的 ID
                })
            # 获取下一页的 URL，如果没有下一页则为 None
            next_page = response.links.get('next', {}).get('url')
            # 若存在下一页，继续请求获取数据
            while next_page is not None:
                response = self.session.get(next_page)
                if response.status_code != 200:
                    logging.error(f"[user] Failed to fetch followers for user: {user_id} from {url}, so far fetched: {len(user_followers)}, status code: {response.status_code}")
                    return user_followers
                for follower in response.json():
                    # self.user_ids.add(follower['user'])
                    user_followers.append({
                        'follower_id': follower['user'],
                        'followee_id': user_id
                    })
                # 更新下一页的 URL
                next_page = response.links.get('next', {}).get('url')
            return user_followers
        except Exception as e:
            # 捕获异常并记录错误日志
            logging.error(f"[user] Exception occurred while fetching followers for user: {user_id}, error: {e}")
            return []
    
    def get_org_followers(self, org_id):
        """
        获取指定组织的所有关注者信息。支持分页获取，若请求失败会记录错误日志。

        Args:
            org_id (str): 组织的 ID。

        Returns:
            list: 包含关注者信息的列表，每个元素是一个字典，包含用户 ID 和组织 ID；
                  若请求失败则返回空列表。
        """
        try:
            # logging.info(f"[org] Processing followers for org: {org_id}")
            # 用于存储组织的关注者信息
            org_followers = []
            # 构建获取组织关注者信息的 API 请求 URL
            url = f"https://huggingface.co/api/organizations/{org_id}/followers"
            # 发送请求获取关注者信息
            response = self.session.get(url)
            # 检查响应状态码，若不是 200 表示请求失败
            if response.status_code != 200:
                logging.error(f"[org] Failed to fetch followers for org: {org_id}, status code: {response.status_code}")
                return []
            # 遍历响应数据，提取关注者信息
            for follower in response.json():
                # self.user_ids.add(follower['user'])
                org_followers.append({
                    'user_id': follower['user'],  # 关注者（用户）的 ID
                    'org_id': org_id  # 被关注的组织 ID
                })
            # 获取下一页的 URL，如果没有下一页则为 None
            next_page = response.links.get('next', {}).get('url')
            # 若存在下一页，继续请求获取数据
            while next_page is not None:
                response = self.session.get(next_page)
                if response.status_code != 200:
                    logging.error(f"[org] Failed to fetch followers for org: {org_id} from {url}, so far fetched: {len(org_followers)}, status code: {response.status_code}")
                    return org_followers
                for follower in response.json():
                    #self.user_ids.add(follower['user'])
                    org_followers.append({
                        'user_id': follower['user'],
                        'org_id': org_id
                    })
                # 更新下一页的 URL
                next_page = response.links.get('next', {}).get('url')
            return org_followers
        except Exception as e:
            # 捕获异常并记录错误日志
            logging.error(f"[org] Exception occurred while fetching followers for org: {org_id}, error: {e}")
            return []
        
    def get_org_members(self, org_id):
        """
        获取指定组织的所有成员信息。支持分页获取，若请求失败会记录错误日志。

        Args:
            org_id (str): 组织的 ID。

        Returns:
            list: 包含成员信息的列表，每个元素是一个字典，包含用户 ID 和组织 ID；
                  若请求失败则返回空列表。
        """
        try:
            # logging.info(f"[org] Processing members for org: {org_id}")
            # 用于存储组织的成员信息
            org_members = []
            # 构建获取组织成员信息的 API 请求 URL
            url = f"https://huggingface.co/api/organizations/{org_id}/members"
            # 发送请求获取成员信息
            response = self.session.get(url)
            # 检查响应状态码，若不是 200 表示请求失败
            if response.status_code != 200:
                logging.error(f"[org] Failed to fetch members for org: {org_id}, status code: {response.status_code}")
                return []
            # 遍历响应数据，提取成员信息
            for member in response.json():
                # self.user_ids.add(member['user'])
                org_members.append({
                    'user_id': member['user'],  # 成员（用户）的 ID
                    'org_id': org_id  # 所属组织的 ID
                })
            # 获取下一页的 URL，如果没有下一页则为 None
            next_page = response.links.get('next', {}).get('url')
            # 若存在下一页，继续请求获取数据
            while next_page is not None:
                response = self.session.get(next_page)
                if response.status_code != 200:
                    logging.error(f"[org] Failed to fetch members for org: {org_id} from {url}, so far fetched: {len(org_members)}, status code: {response.status_code}")
                    return org_members
                for member in response.json():
                    # self.user_ids.add(member['user'])
                    org_members.append({
                        'user_id': member['user'],
                        'org_id': org_id
                    })
                # 更新下一页的 URL
                next_page = response.links.get('next', {}).get('url')
            return org_members
        except Exception as e:
            # 捕获异常并记录错误日志
            logging.error(f"[org] Exception occurred while fetching members for org: {org_id}, error: {e}")
            return []

    def save_user_and_org_data(self):
        """
        保存用户和组织相关的数据到 JSON 文件。
        分别保存处理后的用户、组织数据，以及用户和组织与其他实体的各种关系数据。
        部分数据保存操作被注释，可根据需求取消注释。
        """
        # 保存处理后的用户数据到 'users' 文件
        self.save_data('users', self.processed_users)
        # 保存处理后的组织数据到 'orgs' 文件
        self.save_data('orgs', self.processed_orgs)
        # 保存用户发布模型的关系数据到 'user_publish_model' 文件
        self.save_data('user_publish_model', self.user_publish_model)
        # 保存用户发布数据集的关系数据到 'user_publish_dataset' 文件
        self.save_data('user_publish_dataset', self.user_publish_dataset)
        # 保存用户发布空间的关系数据到 'user_publish_space' 文件
        self.save_data('user_publish_space', self.user_publish_space)
        # 以下数据保存操作被注释，可根据需求取消注释
        # 保存用户发布论文的关系数据到 'user_publish_paper' 文件
        # self.save_data('user_publish_paper', self.user_publish_paper)
        # 保存用户拥有集合的关系数据到 'user_own_collection' 文件
        # self.save_data('user_own_collection', self.user_own_collection)
        # 保存用户点赞模型的关系数据到 'user_like_model' 文件
        # self.save_data('user_like_model', self.user_like_model)
        # 保存用户点赞数据集的关系数据到 'user_like_dataset' 文件
        # self.save_data('user_like_dataset', self.user_like_dataset)
        # 保存用户点赞空间的关系数据到 'user_like_space' 文件
        # self.save_data('user_like_space', self.user_like_space)
        # 保存用户关注用户的关系数据到 'user_follow_user' 文件
        self.save_data('user_follow_user', self.user_follow_user)
        # 保存用户隶属于组织的关系数据到 'user_affiliatedWith_org' 文件
        self.save_data('user_affiliatedWith_org', self.user_affiliatedWith_org)
        # 保存用户关注组织的关系数据到 'user_follow_org' 文件
        self.save_data('user_follow_org', self.user_follow_org)
        # 保存组织发布模型的关系数据到 'org_publish_model' 文件
        self.save_data('org_publish_model', self.org_publish_model)
        # 保存组织发布数据集的关系数据到 'org_publish_dataset' 文件
        self.save_data('org_publish_dataset', self.org_publish_dataset)
        # 保存组织发布空间的关系数据到 'org_publish_space' 文件
        self.save_data('org_publish_space', self.org_publish_space)
        # 保存组织拥有集合的关系数据到 'org_own_collection' 文件，此操作被注释
        # self.save_data('org_own_collection', self.org_own_collection)

    def process_user_and_org_data(self):
        """
        处理用户和组织相关的数据。
        包括对用户名进行分类，获取用户和组织的概览信息、关注者信息、成员信息，
        处理发布关系，最后保存相关数据。
        """
        # 打印开始对用户名进行分类的提示信息
        print(f"Classifying users and organizations...")
        # 使用线程池并发执行用户名分类操作
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 使用 tqdm 显示进度条，对每个用户名调用 username_classification 方法
            list(tqdm(
                executor.map(self.username_classification, self.username_ids),
                total=len(self.username_ids)
            ))
        
        # 打印开始处理组织信息的提示信息
        print(f"Processing organizations...")
        # 使用线程池并发获取组织的概览信息
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(self.get_org_overview, self.org_ids),
                total=len(self.org_ids)
            ))
        # 过滤掉获取失败的结果，保存有效的组织信息
        self.processed_orgs = [org for org in results if org is not None]
        # 打印开始处理组织关注者信息的提示信息
        print(f"Processing followers for organizations...")
        # 使用线程池并发获取组织的关注者信息
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(self.get_org_followers, self.org_ids),
                total=len(self.org_ids)
            ))
        # 将获取到的组织关注者信息添加到对应的列表中
        for result in results:
            self.user_follow_org.extend(result)
        # 打印开始处理组织成员信息的提示信息
        print(f"Processing members for organizations...")
        # 使用线程池并发获取组织的成员信息
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(self.get_org_members, self.org_ids),
                total=len(self.org_ids)
            ))
        # 将获取到的组织成员信息添加到对应的列表中
        for result in results:
            self.user_affiliatedWith_org.extend(result)

        # 打印开始处理用户信息的提示信息
        print(f"Processing users...")
        # 使用线程池并发获取用户的概览信息
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(self.get_user_overview, self.user_ids),
                total=len(self.user_ids)
            ))
        # 过滤掉获取失败的结果，保存有效的用户信息
        self.processed_users = [user for user in results if user is not None]
        # 打印开始处理用户关注者信息的提示信息
        print(f"Processing followers for users...")
        # 使用线程池并发获取用户的关注者信息
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(self.get_user_followers, self.user_ids),
                total=len(self.user_ids)
            ))
        # 将获取到的用户关注者信息添加到对应的列表中
        for result in results:
            self.user_follow_user.extend(result)

        # 打印开始后处理用户和组织发布关系数据的提示信息
        print(f"Post processing user and org data...")
        # 遍历不同类型的发布关系列表
        for edge_list, user_list, org_list, edge_key in [
            (self.username_publish_model, self.user_publish_model, self.org_publish_model, 'model_id'),
            (self.username_publish_dataset, self.user_publish_dataset, self.org_publish_dataset, 'dataset_id'),
            (self.username_publish_space, self.user_publish_space, self.org_publish_space, 'space_id')
        ]:
            for edge in edge_list:
                username = edge['username']
                # 如果用户名是组织 ID，则将发布关系添加到组织发布列表中
                if username in self.org_ids:
                    org_list.append({
                        'org_id': edge['username'],
                        edge_key: edge[edge_key]
                    })
                # 如果用户名是用户 ID，则将发布关系添加到用户发布列表中
                elif username in self.user_ids:
                    user_list.append({
                        'user_id': edge['username'],
                        edge_key: edge[edge_key]
                    })

        # 打印开始保存用户和组织相关数据的提示信息
        print(f"Saving user and org data...")
        # 调用保存方法保存用户和组织相关数据
        self.save_user_and_org_data()

    def run(self):
        """Main execution method"""
        try:
            # 处理任务数据，为后续构建知识图谱准备任务相关信息
            self.process_task_data()
            # 处理模型数据，获取模型标签分类，遍历模型获取详细信息，保存模型数据并记录模型 ID
            self.process_model_data()
            # 多线程批量获取已处理模型的描述信息，并保存更新后的模型数据
            self.process_model_description()
            # 处理数据集数据，获取数据集标签分类，遍历数据集获取详细信息，
            # 保存数据集相关数据，并记录数据集 ID
            self.process_dataset_data()
            # 多线程批量获取已处理数据集的描述信息，并保存更新后的数据集数据
            self.process_dataset_description()
            # 处理 Hugging Face 上的空间数据，获取空间列表，
            # 多线程处理每个空间的详细信息，保存处理结果并记录有效空间 ID
            self.process_space_data()
            # 处理用户对不同类型仓库的点赞数据，多线程获取每种仓库类型下
            # 各个仓库的用户点赞信息，最后保存点赞数据
            self.process_user_like_repo()
            # 处理 Hugging Face 上的集合数据，获取集合标识符列表，
            # 多线程处理每个集合的详细信息，保存处理结果并记录有效集合的标识符
            self.process_collection_data()
            # 处理论文数据，多线程获取每个论文的详细信息，
            # 保存处理结果并记录有效论文的 ID
            self.process_paper_data()
            # 处理用户和组织相关的数据，包括对用户名进行分类，获取用户和组织的概览信息、
            # 关注者信息、成员信息，处理发布关系，最后保存相关数据
            self.process_user_and_org_data()
            # 验证已记录的各种实体之间的关系是否正确
            self.verify_relations()
            # 保存所有处理好的实体数据、关系数据以及额外数据到 JSON 文件
            self.save_all_data()
            # 记录处理数据的统计信息，方便后续分析
            self.log_stats()
            # 从 Hugging Face 平台登出
            logout()
            # 记录数据获取成功完成的日志信息
            logging.info("[constructor] Data fetching completed successfully")
        except Exception as e:
            # 若执行过程中出现异常，记录错误日志，包含异常信息
            logging.error(f"[constructor] Error during execution: {str(e)}")
            # 重新抛出异常，方便上层调用者处理
            raise

def main():
    constructor = KGConstructor()
    constructor.run()

if __name__ == "__main__":
    main()