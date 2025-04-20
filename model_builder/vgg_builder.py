import torch
import torch.nn as nn
import torchvision.models as models
from .base_model_builder import BaseModelBuilder
from .base_model import BaseModelMixin
from .layer_cache import LayerCache
import copy


class BaseVGG(models.vgg.VGG, BaseModelMixin):
    def __init__(self):
        # 使用最小配置初始化
        super().__init__(self.make_layers(
            [64, 'M'], batch_norm=False), init_weights=False)

        # 删除特征层，后续会重新构建
        del self.features

        # 将所有参数转换为bool类型
        self.convert_params_to_bool()

    @staticmethod
    def make_layers(cfg, batch_norm=False):
        """创建VGG的特征层"""
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


class VGGBuilder(BaseModelBuilder):
    def __init__(self):
        self.base_models = nn.ModuleDict()
        self.layer_cache = LayerCache()
        self.init_base_model()

    def init_base_model(self):
        """初始化VGG基础模型"""
        self.base_models['standard'] = BaseVGG()

    def _create_features(self, **kwargs) -> nn.Sequential:
        """创建特征层的工厂函数"""
        cfg = kwargs['cfg']
        batch_norm = kwargs.get('batch_norm', False)

        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

                # 将卷积层参数转换为bool类型
                with torch.no_grad():
                    conv2d.weight.requires_grad_(False)
                    conv2d.weight.data = torch.ones_like(
                        conv2d.weight.data, dtype=torch.bool)
                    if conv2d.bias is not None:
                        conv2d.bias.requires_grad_(False)
                        conv2d.bias.data = torch.ones_like(
                            conv2d.bias.data, dtype=torch.bool)

        return nn.Sequential(*layers)

    def make_layer(self, config: list, **kwargs) -> nn.Module:
        """实现抽象方法 make_layer"""
        return self.layer_cache.get_or_create(
            cache_name='features',
            creator_fn=self._create_features,
            cfg=config,
            batch_norm=kwargs.get('batch_norm', False)
        )

    def get_model(self, model_name: str) -> nn.Module:
        """获取VGG模型"""
        configs = {
            'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }

        if model_name not in configs:
            raise ValueError(f"Unsupported VGG model: {model_name}")

        model = copy.deepcopy(self.base_models['standard'])
        model.features = self.make_layer(configs[model_name])
        model.convert_params_to_bool()
        return model
