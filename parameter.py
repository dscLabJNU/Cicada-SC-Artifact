import hashlib
import time
from collections import OrderedDict
import os
import torch
import tensorly as tl
from typing import Optional
from logger_config import get_logger

logger = get_logger(__name__)


def get_tensor_size_in_bytes(tensor):
    """
    计算张量所占内存字节数的辅助函数。
    """
    return tensor.element_size() * tensor.numel()


class StorageBackend:
    """
    存储后端基类，定义必要的接口。
    """

    def store_parameter(self, param_id, param_tensor):
        raise NotImplementedError

    def get_parameter(self, param_id):
        raise NotImplementedError

    def has_parameter(self, param_id):
        """检查参数是否存在（轻量级检查）"""
        raise NotImplementedError

    def load_existing_parameters(self):
        """
        加载已有的参数（如果有的话）。
        """
        pass


class InMemoryStorage(StorageBackend):
    """
    内存存储后端，实现基于内存的参数缓存管理。
    """

    def __init__(self, max_size=2 * 1024 * 1024 * 1024):
        self.max_size = max_size
        self.current_size = 0
        self.parameters = OrderedDict()
        self.eviction_callbacks = []  # 添加回调列表
        logger.info(
            f"Initializing InMemoryStorage with max_size: {max_size/1024/1024:.2f}MB")

    def add_eviction_callback(self, callback):
        """添加参数逐出时的回调函数"""
        self.eviction_callbacks.append(callback)

    def ensure_cache_size(self):
        """
        确保当前缓存大小不超过 max_size。
        超出时逐出最久未使用的参数。
        """
        logger.debug(
            f"Current cache size: {self.current_size / 1024 / 1024:.2f} MB."
        )
        while self.current_size > self.max_size and self.parameters:
            oldest_key, oldest_tensor = self.parameters.popitem(last=False)
            oldest_size = get_tensor_size_in_bytes(oldest_tensor)
            self.current_size -= oldest_size

            # 触发所有回调
            for callback in self.eviction_callbacks:
                callback(oldest_key, oldest_tensor)

    def store_parameter(self, param_id, param_tensor):
        """
        存储参数数据，如果已存在则复用并更新LRU顺序。
        """
        if param_id not in self.parameters:
            param_size = get_tensor_size_in_bytes(param_tensor)
            self.current_size += param_size
            self.ensure_cache_size()
            self.parameters[param_id] = param_tensor.detach().clone()
            logger.debug(
                f"Stored new parameter with ID: {param_id}"
            )
        else:
            logger.debug(
                f"Reusing existing parameter with ID: {param_id}"
            )

        # 将此参数标记为最近使用
        self.parameters.move_to_end(param_id, last=True)

    def get_parameter(self, param_id):
        """
        根据 param_id 获取参数并将其标记为最新使用。
        如果不存在则返回 None。
        """
        param_tensor = self.parameters.get(param_id)
        if param_tensor is not None:
            self.parameters.move_to_end(param_id, last=True)
        return param_tensor

    def has_parameter(self, param_id):
        """轻量级检查参数是否存在"""
        return param_id in self.parameters


class PersistentStorage(StorageBackend):
    """
    持久化存储后端，实现将参数存储到磁盘。
    """

    def __init__(self, directory):
        self.directory = directory
        logger.info(f"Initializing PersistentStorage at: {directory}")
        os.makedirs(self.directory, exist_ok=True)
        self.param_dimensions = {}  # 存储 param_id 对应维度的字典
        self.load_existing_parameters()
        self.compress_suffix = "_tucker"

    def _get_param_filepath(self, param_id, dimensions, compress=False):
        """
        根据 param_id 获取参数文件的完整路径，包含张量维度作为目录后缀。
        """
        if not dimensions:
            dim_suffix = "0"
        else:
            dim_suffix = "_".join(str(dim) for dim in dimensions)

        if compress:
            dim_suffix += self.compress_suffix

        directory_with_dims = os.path.join(self.directory, dim_suffix)
        os.makedirs(directory_with_dims, exist_ok=True)
        return os.path.join(directory_with_dims, f"{param_id}.pt")

    def store_parameter(self, param_id, param_tensor):
        """
        存储参数到磁盘。
        """
        dimensions = param_tensor.shape
        self.param_dimensions[param_id] = dimensions  # 存储 param_id 对应的维度
        filepath = self._get_param_filepath(param_id, dimensions)
        try:
            torch.save(param_tensor.detach().clone(), filepath)
            logger.debug(
                f"Stored new parameter with ID: {param_id} to {filepath}"
            )
        except Exception as e:
            logger.error(
                f"Failed to store parameter {param_id} to disk: {e}"
            )

    def get_parameter(self, param_id):
        """
        从磁盘获取参数。如果是 Tucker 压缩格式，自动重构为张量。
        """
        dimensions = self.param_dimensions.get(param_id)
        if dimensions is None:
            logger.debug(f"未找到参数 {param_id} 的维度信息。")
            return None

        # 首先检查是否存在 tucker 后缀的目录
        filepath = self._get_param_filepath(
            param_id, dimensions, compress=False)

        # 如果 tucker 后缀的目录不存在，则检查普通目录
        if not os.path.exists(filepath):
            filepath = self._get_param_filepath(param_id, dimensions)
            if not os.path.exists(filepath):
                logger.debug(f"参数文件 {filepath} 不存在。")
                return None
        try:
            # compressed_data = torch.load(filepath, map_location='cpu', weights_only=True)
            compressed_data = torch.load(filepath)

            # 检查是否是 Tucker 压缩格式并重构
            if isinstance(compressed_data, dict) and 'core' in compressed_data and 'factors' in compressed_data:
                logger.debug(f"检测到 Tucker 压缩格式，正在重构张量...")
                reconstructed_tensor = tl.tucker_to_tensor((
                    compressed_data['core'],
                    compressed_data['factors']
                ))
                tensor = torch.from_numpy(reconstructed_tensor)
            else:
                tensor = compressed_data

            logger.debug(
                f"Loaded parameter with ID: {param_id} from {filepath}"
            )
            return tensor
        except Exception as e:
            logger.error(
                f"Failed to load parameter {param_id} from disk: {e}"
            )
            return None

    def load_existing_parameters(self):
        """
        加载目录中已有的参数文件到日志中（可选）。
        """
        logger.debug(f"Loading existing parameters from {self.directory}")
        for dim_dir in os.listdir(self.directory):
            dim_dir_path = os.path.join(self.directory, dim_dir)
            if os.path.isdir(dim_dir_path):
                if "tucker" in dim_dir_path:
                    continue
                dims = dim_dir.split('_')
                try:
                    dimensions = tuple(int(dim)
                                       for dim in dims)  # 假设目录格式为 dim1_dim2
                except ValueError:
                    logger.error(
                        f"目录 {dim_dir} 的维度格式无效。"
                    )
                    continue
                for filename in os.listdir(dim_dir_path):
                    if filename.endswith(".pt"):
                        param_id = filename[:-3]
                        self.param_dimensions[param_id] = dimensions
                        filepath = os.path.join(dim_dir_path, filename)
                        try:
                            _ = torch.load(
                                filepath, map_location='cpu', weights_only=True)
                            # logger.debug(
                            #     f"Loaded existing parameter with ID: {param_id}"
                            # )
                        except Exception as e:
                            logger.error(
                                f"Failed to load existing parameter {param_id}: {e}"
                            )

    def has_parameter(self, param_id):
        """轻量级检查参数文件是否存在"""
        dimensions = self.param_dimensions.get(param_id)
        if dimensions is None:
            return False

        filepath = self._get_param_filepath(param_id, dimensions)
        return os.path.exists(filepath)


class HybridStorage(StorageBackend):
    """混合存储后端，结合内存和磁盘存储"""

    def __init__(self, max_size=2 * 1024 * 1024 * 1024, backup_dir='./evicted_params'):
        self.memory_storage = InMemoryStorage(max_size)
        self.disk_storage = PersistentStorage(backup_dir)
        logger.info(
            f"Initializing HybridStorage with max_size: {max_size/1024/1024:.2f}MB")
        logger.info(f"Backup directory: {backup_dir}")

    def store_parameter(self, param_id, param_tensor):
        """存储参数到内存"""
        self.memory_storage.store_parameter(param_id, param_tensor)

    def get_parameter(self, param_id):
        """
        获取参数，优先从内存获取，如果内存中不存在则尝试从磁盘加载
        """
        param_tensor = self.memory_storage.get_parameter(param_id)
        if param_tensor is None and self.disk_storage.has_parameter(param_id):
            logger.info(
                f"Parameter {param_id} not in memory, loading from disk...")
            param_tensor = self.disk_storage.get_parameter(param_id)
            if param_tensor is not None:
                # 重新加入内存缓存
                self.memory_storage.store_parameter(param_id, param_tensor)
        return param_tensor

    def has_parameter(self, param_id):
        """检查参数是否存在于内存或磁盘中"""
        return self.memory_storage.has_parameter(param_id) or self.disk_storage.has_parameter(param_id)

    def _handle_eviction(self, param_id, param_tensor):
        """处理参数被逐出时的回调"""
        logger.info(
            f"Parameter {param_id} evicted from memory, saving to disk...")
        self.disk_storage.store_parameter(param_id, param_tensor)


class ParameterCache:
    """
    参数缓存类，实现对参数的缓存管理，支持内存和持久化存储。
    """

    def __init__(self, max_size: int = 2 * 1024 * 1024 * 1024, persistent: bool = False):
        """
        初始化参数缓存。

        参数：
        - max_size (int): 最大缓存大小（仅适用于内存缓存）。
        - persistent (str or None): 如果提供，指向一个目录路径，用于持久化存储参数。
        """
        logger.debug(f"max_size: {max_size}, persistent: {persistent}")
        
        # 创建日志目录
        log_dir = './logs/tetris'
        os.makedirs(log_dir, exist_ok=True)
        
        # 初始化CSV日志文件
        self.param_log = open(f'{log_dir}/memory_stats.csv', 'w')
        
        # 写入CSV表头
        print("timestamp,param_id,param_size_mb,total_mb", 
              file=self.param_log, flush=True)
        
        # 跟踪累计内存使用
        self.cumulative_size = 0
        
        if persistent:
            logger.debug(
                f"Initializing PersistentStorage with directory: {persistent}")
            self.storage_backend = PersistentStorage(persistent)
            self.is_persistent = True
        else:
            self.storage_backend = HybridStorage(max_size)  # 使用混合存储
            if isinstance(self.storage_backend, HybridStorage):
                self.storage_backend.memory_storage.add_eviction_callback(
                    self.storage_backend._handle_eviction
                )
            self.is_persistent = False

    def __del__(self):
        """确保文件正确关闭"""
        self.param_log.close()

    def _get_tensor_size_mb(self, tensor: torch.Tensor) -> float:
        """计算张量的内存占用（MB）"""
        return tensor.element_size() * tensor.nelement() / (1024 * 1024)

    def store_parameter(self, param_id: str, param_tensor: torch.Tensor) -> None:
        """
        存储参数到缓存（内存或磁盘）。

        参数：
        - param_id (str): 参数的唯一标识符。
        - param_tensor (torch.Tensor): 参数张量。
        """
        # 计算参数大小
        param_size_mb = self._get_tensor_size_mb(param_tensor)
        self.cumulative_size += param_size_mb
        
        # 记录到日志
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"{timestamp},{param_id},{param_size_mb:.2f},{self.cumulative_size:.2f}", 
              file=self.param_log, flush=True)
        
        # 存储参数
        self.storage_backend.store_parameter(param_id, param_tensor)

    def get_parameter(self, param_id: str) -> Optional[torch.Tensor]:
        """
        从缓存获取参数。

        参数：
        - param_id (str): 参数的唯一标识符。

        返回：
        - Optional[torch.Tensor]: 参数张量，如果未找到则返回 None。
        """
        # 直接调用存储后端的 get_parameter 方法
        return self.storage_backend.get_parameter(param_id)

    def compute_param_id(self, param_name: str, model_name: str = "") -> str:
        """
        根据参数名称和模型名称生成唯一的 param_id。

        参数：
        - param_name (str): 参数名称
        - model_name (str): 模型名称，用于确保不同模型的同名参数不会冲突

        返回：
        - str: 生成的 param_id
        """
        # 将模型名称和参数名称组合，确保唯一性
        unique_key = f"{model_name}:{param_name}"
        return hashlib.md5(unique_key.encode('utf-8')).hexdigest()

    def has_parameter(self, param_id: str) -> bool:
        """
        轻量级检查参数是否存在。
        不会加载实际的参数内容。
        """
        return self.storage_backend.has_parameter(param_id)

if __name__ == "__main__":
    import time
    from logger_config import get_logger
    from model_loader import load_model_from_source, ModelConfig
    import utils
    import os

    logger = get_logger(__name__)

    # 加载 YAML 配置
    yaml_path = 'models_config.yaml'
    model_config = ModelConfig(yaml_path)
    model_fn_map = model_config.get_model_fn_map()
    weights_fn_map = model_config.get_weights_fn_map()

    # 初始化参数缓存（4GB 内存缓存）
    cache = ParameterCache(max_size=40 * 1024 * 1024 * 1024)
    
    # 定义要测试的模型配置
    MODELS_TO_TEST = [
        # Vision Transformer models
        {"model_name": "vit_b_16_IMAGENET1K_V1", "weights_key": "ViT_B_16_Weights.IMAGENET1K_V1"},
        {"model_name": "vit_b_16_default", "weights_key": "ViT_B_16_Weights.DEFAULT"},
        {"model_name": "vit_b_32_default", "weights_key": "ViT_B_32_Weights.DEFAULT"},
        {"model_name": "vit_l_16_default", "weights_key": "ViT_L_16_Weights.DEFAULT"},
        
        # ResNet models
        {"model_name": "resnet18_default", "weights_key": "ResNet18_Weights.DEFAULT"},
        {"model_name": "resnet34_default", "weights_key": "ResNet34_Weights.DEFAULT"},
        {"model_name": "resnet50_default", "weights_key": "ResNet50_Weights.DEFAULT"},
        {"model_name": "resnet101_default", "weights_key": "ResNet101_Weights.DEFAULT"},
        {"model_name": "resnet152_default", "weights_key": "ResNet152_Weights.DEFAULT"},
        
        # VGG models
        {"model_name": "vgg11_default", "weights_key": "VGG11_Weights.DEFAULT"},
        {"model_name": "vgg13_default", "weights_key": "VGG13_Weights.DEFAULT"},
        {"model_name": "vgg16_default", "weights_key": "VGG16_Weights.DEFAULT"},
        {"model_name": "vgg19_default", "weights_key": "VGG19_Weights.DEFAULT"},
        
        # DenseNet models
        {"model_name": "densenet121_default", "weights_key": "DenseNet121_Weights.DEFAULT"},
        {"model_name": "densenet169_default", "weights_key": "DenseNet169_Weights.DEFAULT"},
        {"model_name": "densenet201_default", "weights_key": "DenseNet201_Weights.DEFAULT"},
        {"model_name": "densenet161_default", "weights_key": "DenseNet161_Weights.DEFAULT"},
        
        # MobileNet models
        {"model_name": "mobilenet_v2_default", "weights_key": "MobileNet_V2_Weights.DEFAULT"},
        {"model_name": "mobilenet_v3_large_default", "weights_key": "MobileNet_V3_Large_Weights.DEFAULT"},
        {"model_name": "mobilenet_v3_small_default", "weights_key": "MobileNet_V3_Small_Weights.DEFAULT"},
        
        # EfficientNet models
        {"model_name": "efficientnet_b0_default", "weights_key": "EfficientNet_B0_Weights.DEFAULT"},
        {"model_name": "efficientnet_b1_default", "weights_key": "EfficientNet_B1_Weights.DEFAULT"},
        
        # SqueezeNet models
        {"model_name": "squeezenet1_0_default", "weights_key": "SqueezeNet1_0_Weights.DEFAULT"},
        {"model_name": "squeezenet1_1_default", "weights_key": "SqueezeNet1_1_Weights.DEFAULT"},
        
        # GoogLeNet model
        {"model_name": "googlenet_default", "weights_key": "GoogLeNet_Weights.DEFAULT"},
        
        # AlexNet model
        {"model_name": "alexnet_default", "weights_key": "AlexNet_Weights.DEFAULT"},
    ]

    # 设置输出文件
    output_file = os.getenv('PARAMETER_CACHE_OUTPUT', 'logs/parameter_cache/model_load_test.csv')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w") as log_file:
        print("model_name,operation,time", file=log_file, flush=True)
        
        # 存储阶段
        for model_info in MODELS_TO_TEST:
            model_name = model_info["model_name"]
            weights_key = model_info["weights_key"]
            
            print("="*40)
            print(f"\nProcessing model '{model_name}' with weights '{weights_key}'...")
            
            try:
                # 加载原始模型
                time_start = time.time()
                model_original = load_model_from_source(model_name, weights_key)
                time_load = time.time() - time_start
                logger.info(f"加载原始模型 {model_name} 用时: {time_load:.2f}秒")
                print(f"{model_name},load,{time_load}", file=log_file, flush=True)

                if model_original is None:
                    logger.error(f"无法加载模型 {model_name}")
                    continue

                # 存储模型参数
                time_start = time.time()
                for name, param in model_original.state_dict().items():
                    param_id = cache.compute_param_id(name, model_name)
                    cache.store_parameter(param_id, param)
                time_store = time.time() - time_start
                logger.info(f"缓存模型 {model_name} 参数用时: {time_store:.2f}秒")
                print(f"{model_name},store,{time_store}", file=log_file, flush=True)

                # 恢复阶段
                time_start = time.time()
                new_model = load_model_from_source(model_name, "")  # 创建无权重模型
                
                # 从缓存恢复权重
                new_state_dict = {}
                for name in model_original.state_dict().keys():
                    param_id = cache.compute_param_id(name, model_name)
                    param = cache.get_parameter(param_id)
                    if param is not None:
                        new_state_dict[name] = param
                    else:
                        logger.error(f"无法从缓存加载参数: {name}")

                # 加载权重
                new_model.load_state_dict(new_state_dict, strict=False)
                time_restore = time.time() - time_start
                logger.info(f"从缓存加载模型 {model_name} 用时: {time_restore:.2f}秒")
                print(f"{model_name},restore,{time_restore}", file=log_file, flush=True)

                # 验证模型参数一致性
                logger.info(f"验证模型 {model_name} 参数一致性:")
                start = time.time()
                is_valid = utils.verify_model_state_dict(
                    new_model.state_dict(),
                    model_original.state_dict(),
                    model_name
                )
                time_verify = time.time() - start
                print(f"{model_name},verify,{time_verify}", file=log_file, flush=True)

                if is_valid:
                    logger.info(f"所有参数完全匹配! 验证用时: {time_verify:.2f}秒")
                else:
                    logger.warning(f"存在参数不匹配! 验证用时: {time_verify:.2f}秒")

            except Exception as e:
                logger.error(f"处理模型 {model_name} 时出错: {str(e)}")
                continue

            print("="*40)
            print("\n")
