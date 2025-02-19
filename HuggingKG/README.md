# HuggingKG

Code for constructing HuggingKG.

## How to Run

1. Install requirements:
    ```
    pip install -r requirements.txt
    ```
2. (Optional) Configure your Hugging Face token in "HuggingKG_constructor.py" to increase API rate limits:
    ```python
    class KGConstructor:
    ···
    def setup_environment(self):
        """Setup environment variables and logging"""
        self.my_hf_token = "" # Optional: Add your Hugging Face API token here to increase rate limits
        ···
    ···
    ```
3. Run the script:
    ```
    python HuggingKG_constructor.py
    ```

## Statistics

(collected on December 15, 2024)

The entire process takes approximately 20 hours and the storage of node attributes and edges in the graph amounts to around 5.8 GB.

The current version of HuggingKG is available on [Hugging Face](https://huggingface.co/datasets/cqsss/HuggingKG).

| entity       | total   |
| ------------ | ------- |
| task         | 52      |
| model        | 1202732 |
| dataset      | 261663  |
| space        | 307789  |
| paper        | 15857   |
| collection   | 79543   |
| user         | 729401  |
| organization | 17233   |
| （all）      | 2614270 |

| subject      | predicate                | object       | total   |
| ------------ | ------------------------ | ------------ | ------- |
| model        | defined  for             | task         | 532391  |
| model        | adapter                  | model        | 155642  |
| model        | finetune                 | model        | 107162  |
| model        | merge                    | model        | 20003   |
| model        | quantized                | model        | 45809   |
| model        | trained  or finetuned on | dataset      | 96546   |
| model        | cite                     | paper        | 272455  |
| dataset      | defined  for             | task         | 38193   |
| dataset      | cite                     | paper        | 10411   |
| space        | use                      | model        | 312904  |
| space        | use                      | dataset      | 5071    |
| collection   | contain                  | model        | 134515  |
| collection   | contain                  | dataset      | 40771   |
| collection   | contain                  | space        | 53833   |
| collection   | contain                  | paper        | 42980   |
| user         | publish                  | model        | 1073429 |
| user         | publish                  | dataset      | 197226  |
| user         | publish                  | space        | 294808  |
| user         | publish                  | paper        | 25999   |
| user         | own                      | collection   | 74089   |
| user         | like                     | model        | 1144281 |
| user         | like                     | dataset      | 236419  |
| user         | like                     | space        | 586316  |
| user         | follow                   | user         | 306135  |
| user         | follow                   | organization | 170232  |
| user         | affiliated  with         | organization | 57220   |
| organization | publish                  | model        | 128804  |
| organization | publish                  | dataset      | 64300   |
| organization | publish                  | space        | 12956   |
| organization | own                      | collection   | 5453    |
|              | (all)                    |              | 6246353 |

## Building the Knowledge Graph

The main workflow is managed by the `run` method, sequentially executing data collection, validation, and storage. Multi-threading is used in multiple steps to speed up data processing.

### Preparation
The constructor first initializes the runtime environment, including logging in with the API token provided by Hugging Face, setting up an output directory to store the generated data, and configuring a logging system to record detailed execution information. To ensure stability during execution, we sets up an HTTP session, incorporating retry mechanisms and request timeout controls.

### Data Crawling and Conversion

Processed entities and relationships are stored as JSON files for future use.

Each type of entity and relationship is saved separately, along with auxiliary data such as ID sets to ensure data integrity.

| Entities     | Sources/Methods                                              | Notes                                       |
| ------------ | ------------------------------------------------------------ | ------------------------------------------- |
| Task         | `/api/tasks`, <br />'pipeline_tag' tag found in Model data,<br />'task_categories' tag found in Dataset data |                                             |
| Model        | `list_models`(huggingface_hub), <br />`/{model.id}/resolve/main/README.md` as ‘description’ | the maximum size of 'description' is 100 MB |
| Dataset      | `list_datasets`(huggingface_hub), <br />`/{dataset.id}/resolve/main/README.md` as ‘description’ | the maximum size of 'description' is 100 MB |
| Space        | `list_spaces`(huggingface_hub) for a list of Space, <br />`/api/spaces/{space.id}` for details |                                             |
| Collection   | `/api/collections`  for a list of Collection, <br />`/api/collections/{collection.slug}` for details |                                             |
| Paper        | 'arxiv:' tag found in Model data and Dataset data, <br />'item' with 'type:paper' found in Collection data, <br />`api/papers/{arxiv_id}` for details |                                             |
| User         | 'author' found in Model data, Dataset data, Space data and Paper data, <br />'owner' found in Collection data, <br />`/users/{user.id}/overview` for details |                                             |
| Organization | 'author' found in Model data, Dataset data and Space data, <br />'owner' found in Collection data, <br />`/api/organizations/{organization.id}/overview` for details |                                             |

| Relations                                         | Sources/Methods                                              | Notes |
| ------------------------------------------------- | ------------------------------------------------------------ | ----- |
| Model - Defined For - Task                        | 'pipeline_tag:' tag found in Model data                      |       |
| Model - Adapter/Finetune/Merge/Quantize - Model   | 'base_model:' tag found in Model data                        |       |
| Model - Trained Or Finetuned On - Dataset         | 'dataset:' tag found in Model data                           |       |
| Model/Dataset - Cite - Paper                      | 'arxiv:' tag found in Model data and Dataset data            |       |
| Dataset - Defined For - Task                      | 'task_categories' tag found in Dataset data                  |       |
| Space - Use - Model/Dataset                       | 'models' and 'datasets' found in Space data                  |       |
| Collection - Contain - Model/Dataset/Space/Paper  | 'items' found in Collection data                             |       |
| User/Organization - Publish - Model/Dataset/Space | 'author' found in Model data, Dataset data and Space data    |       |
| User - Publish - Paper                            | 'authors' found in Paper Data                                |       |
| User/Organization - Own - Collection              | 'owner' found in Collection data                             |       |
| User - Like - Model/Dataset/Space                 | `/api/{repo_type}s/{repo_id}/likers`                         |       |
| User - Follow - User/Organization                 | `/api/users/{user_id}/followers`                             |       |
| User - Affiliated With - Organization             | `/api/organizations/{org_id}/members`, <br />`/api/organizations/{org_id}/followers` |       |

### Data Verification and Cleaning

After collecting all data, the `verify_relations` method validates relationships by ensuring referenced IDs exist in the corresponding entity lists. Invalid relationships are logged and filtered.
