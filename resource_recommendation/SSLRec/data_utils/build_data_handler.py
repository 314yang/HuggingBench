from config.configurator import configs
import importlib

def build_data_handler():
    """
    根据配置文件中的数据类型动态导入并实例化对应的 DataHandler 类。

    Returns:
        DataHandler 类的实例。

    Raises:
        NotImplementedError: 如果指定的 DataHandler 模块未实现或类未定义，抛出此异常。
    """
    # 根据配置文件中的数据类型构建 DataHandler 模块名
    datahandler_name = 'data_handler_' + configs['data']['type']
    # 构建模块的完整路径
    module_path = ".".join(['data_utils', datahandler_name])
    # 检查指定路径的模块是否存在
    if importlib.util.find_spec(module_path) is None:
        # 若模块不存在，抛出异常提示未实现该 DataHandler
        raise NotImplementedError('DataHandler {} is not implemented'.format(datahandler_name))
    # 导入指定路径的模块
    module = importlib.import_module(module_path)
    # 遍历模块中的所有属性
    for attr in dir(module):
        # 比较属性名（转换为小写并去除下划线）与 DataHandler 名称是否一致
        if attr.lower() == datahandler_name.lower().replace('_', ''):
            # 若一致，获取该属性并实例化后返回
            return getattr(module, attr)()
    else:
        # 若未找到匹配的类，抛出异常提示在指定模块中未定义该 DataHandler 类
        raise NotImplementedError('DataHandler Class {} is not defined in {}'.format(datahandler_name, module_path))

# 以下是旧版本的 build_data_handler 函数，已注释
# def build_data_handler():
#     if configs['data']['type'] == 'general_cf':
#         data_handler = DataHandlerGeneralCF()
#     elif configs['data']['type'] == 'sequential':
#         data_handler = DataHandlerSequential()
#     elif configs['data']['type'] == 'multi_behavior':
#         data_handler = DataHandlerMultiBehavior()
#     elif configs['data']['type'] == 'social':
#         data_handler = DataHandlerSocial()
#     elif configs['data']['type'] == 'kg':
#         data_handler = DataHandlerKG()
#     else:
#         raise NotImplementedError

#     return data_handler