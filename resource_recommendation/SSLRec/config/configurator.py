import os
import yaml
import argparse

def parse_configure():
    """
    解析命令行参数并加载模型配置文件，合并配置信息。

    此函数会先解析命令行参数，根据参数设置 CUDA 可见设备，
    然后加载对应模型的 YAML 配置文件，对配置进行补充和校验，最后返回完整的配置信息。

    Raises:
        Exception: 当未提供模型名或模型对应的 YAML 文件不存在，
                   或配置中 'patience' 值小于等于 0 时抛出。

    Returns:
        dict: 包含完整配置信息的字典。
    """
    # 创建命令行参数解析器，设置描述信息
    parser = argparse.ArgumentParser(description='SSLRec')
    # 添加 --model 参数，用于指定模型名称
    parser.add_argument('--model', type=str, help='Model name')
    # 添加 --dataset 参数，用于指定数据集名称，默认值为 None
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name')
    # 添加 --device 参数，用于指定使用的设备，默认值为 'cuda'
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    # 添加 --cuda 参数，用于指定 CUDA 设备编号，默认值为 '0'
    parser.add_argument('--cuda', type=str, default='0', help='Device number')
    # 解析命令行参数
    args = parser.parse_args()

    # 如果使用的设备是 cuda，设置 CUDA 可见设备编号
    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    # 检查是否提供了模型名，若未提供则抛出异常
    if args.model is None:
        raise Exception("Please provide the model name through --model.")
    # 将模型名转换为小写
    model_name = args.model.lower()
    # 检查模型对应的 YAML 配置文件是否存在，若不存在则抛出异常
    if not os.path.exists('./config/modelconf/{}.yml'.format(model_name)):
        raise Exception("Please create the yaml file for your model first.")

    # 打开模型对应的 YAML 配置文件
    with open('./config/modelconf/{}.yml'.format(model_name), encoding='utf-8') as f:
        # 读取文件内容
        config_data = f.read()
        # 使用 yaml 库解析文件内容，得到配置信息
        configs = yaml.safe_load(config_data)

        # 将配置中的模型名转换为小写
        configs['model']['name'] = configs['model']['name'].lower()

        # 如果配置中没有 'tune' 字段，添加该字段并设置其 'enable' 为 False
        if 'tune' not in configs:
            configs['tune'] = {'enable': False}

        # 将命令行指定的设备信息添加到配置中
        configs['device'] = args.device

        # 如果命令行指定了数据集名称，更新配置中的数据集名称
        if args.dataset is not None:
            configs['data']['name'] = args.dataset

        # 如果配置的训练部分没有 'log_loss' 字段，添加该字段并设置为 True
        if 'log_loss' not in configs['train']:
            configs['train']['log_loss'] = True

        # 处理早停相关配置
        if 'patience' in configs['train']:
            # 检查 'patience' 值是否小于等于 0，若小于等于 0 则抛出异常
            if configs['train']['patience'] <= 0:
                raise Exception("'patience' should be greater than 0.")
            else:
                # 若 'patience' 值合法，设置早停标志为 True
                configs['train']['early_stop'] = True
        else:
            # 若配置中没有 'patience' 字段，设置早停标志为 False
            configs['train']['early_stop'] = False

        return configs

# 调用 parse_configure 函数获取配置信息并存储在 configs 变量中
configs = parse_configure()