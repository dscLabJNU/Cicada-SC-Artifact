import torch


class BaseModelMixin:
    """基础模型Mixin，提供通用的参数转换方法"""

    def convert_params_to_bool(self):
        """将所有参数转换为bool类型"""
        with torch.no_grad():
            for param in self.parameters():
                param.requires_grad_(False)
                param.data = torch.ones_like(param.data, dtype=torch.bool)
