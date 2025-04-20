import torch
import torch.nn as nn
import torchvision.models as models
from .base_model_builder import BaseModelBuilder
from .base_model import BaseModelMixin
from .layer_cache import LayerCache
import copy


class BaseResNet(models.resnet.ResNet, BaseModelMixin):
    def __init__(self, block):
        # 调用父类的 __init__，但使用最小配置
        super().__init__(block, [1, 1, 1, 1], init_weights=False)
        # 将所有参数转换为 bool 类型
        self.convert_params_to_bool()


class ResNetBuilder(BaseModelBuilder):
    def __init__(self):
        super().__init__()
        self.base_models = nn.ModuleDict()
        self.layer_cache = LayerCache()
        self.init_base_model()

    def init_base_model(self):
        """初始化ResNet基础模型"""
        self.base_models['basic'] = BaseResNet(models.resnet.BasicBlock)
        self.base_models['bottleneck'] = BaseResNet(models.resnet.Bottleneck)

    def _create_downsample(self, **kwargs) -> nn.Sequential:
        """创建下采样层的工厂函数"""
        norm_layer = kwargs['norm_layer']
        inplanes = kwargs['inplanes']
        planes = kwargs['planes']
        stride = kwargs['stride']
        expansion = kwargs['expansion']

        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * expansion,
                      kernel_size=1, stride=stride, bias=False),
            norm_layer(planes * expansion),
        )

        return downsample

    def _create_block(self, **kwargs) -> nn.Module:
        """创建 ResNet block 的工厂函数"""
        block_type = kwargs['block_type']
        inplanes = kwargs['inplanes']
        planes = kwargs['planes']
        stride = kwargs.get('stride', 1)
        groups = kwargs.get('groups', 1)
        base_width = kwargs.get('base_width', 64)
        dilation = kwargs.get('dilation', 1)
        norm_layer = kwargs['norm_layer']

        # 确定是否需要下采样
        expansion = 4 if block_type == models.resnet.Bottleneck else 1
        if stride != 1 or inplanes != planes * expansion:
            downsample = self.layer_cache.get_or_create(
                cache_name='downsample',
                creator_fn=self._create_downsample,
                norm_layer=norm_layer,
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                expansion=expansion
            )
        else:
            downsample = None

        block = block_type(
            inplanes, planes, stride, downsample,
            groups, base_width, dilation, norm_layer
        )

        return block

    def make_layer(self, config: dict, **kwargs) -> nn.Module:
        """实现抽象方法 make_layer"""
        base_model = kwargs['base_model']
        block_type = kwargs['block_type']
        planes = config['planes']
        blocks = config['blocks']
        stride = config.get('stride', 1)

        layers = []
        # 创建第一个 block
        first_block_config = {
            'block_type': block_type,
            'inplanes': base_model.inplanes,
            'planes': planes,
            'stride': stride,
            'groups': base_model.groups,
            'base_width': base_model.base_width,
            'dilation': base_model.dilation,
            'norm_layer': base_model._norm_layer
        }

        first_block = self.layer_cache.get_or_create(
            cache_name=f'block_{planes}_{stride}',
            creator_fn=self._create_block,
            **first_block_config
        )
        layers.append(first_block)

        # 更新 inplanes
        base_model.inplanes = planes * block_type.expansion

        # 创建剩余的 blocks
        for i in range(1, blocks):
            block_config = {
                'block_type': block_type,
                'inplanes': base_model.inplanes,
                'planes': planes,
                'groups': base_model.groups,
                'base_width': base_model.base_width,
                'dilation': base_model.dilation,
                'norm_layer': base_model._norm_layer
            }

            block = self.layer_cache.get_or_create(
                cache_name=f'block_{planes}_1_{i}',
                creator_fn=self._create_block,
                **block_config
            )
            layers.append(block)

        return nn.Sequential(*layers)

    def get_model(self, model_name: str) -> nn.Module:
        """获取ResNet模型"""
        configs = {
            'resnet18': ([2, 2, 2, 2], 'basic', 64),
            'resnet34': ([3, 4, 6, 3], 'basic', 64),
            'resnet50': ([3, 4, 6, 3], 'bottleneck', 64),
            'resnet101': ([3, 4, 23, 3], 'bottleneck', 64),
            'resnet152': ([3, 8, 36, 3], 'bottleneck', 64),
            # base_width 加倍
            'wide_resnet50_2': ([3, 4, 6, 3], 'bottleneck', 64 * 2),
            # base_width 加倍
            'wide_resnet101_2': ([3, 4, 23, 3], 'bottleneck', 64 * 2)
        }

        if model_name not in configs:
            raise ValueError(f"Unsupported ResNet model: {model_name}")

        layers, model_type, base_width = configs[model_name]
        model = copy.deepcopy(self.base_models[model_type])
        block_type = models.resnet.BasicBlock if model_type == 'basic' else models.resnet.Bottleneck

        # 设置 base_width
        model.base_width = base_width
        model.inplanes = 64

        in_planes = 64
        planes = [64, 128, 256, 512]
        strides = [1, 2, 2, 2]

        for i, num_blocks in enumerate(layers):
            layer_config = {
                'planes': planes[i],
                'blocks': num_blocks,
                'stride': strides[i]
            }

            layer = self.make_layer(
                config=layer_config,
                base_model=model,
                block_type=block_type
            )

            setattr(model, f'layer{i+1}', layer)

        model.convert_params_to_bool()
        return model
