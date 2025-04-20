import torch
import torch.nn as nn
import torchvision.models as models
from .base_model_builder import BaseModelBuilder
from .base_model import BaseModelMixin
from .layer_cache import LayerCache
import copy
from collections import OrderedDict
from functools import partial
import math
from typing import Dict, Optional
from torchvision.models.vision_transformer import MLPBlock, EncoderBlock, Encoder


class BaseViT(models.VisionTransformer, BaseModelMixin):
    def __init__(self):
        # 使用最小配置初始化，减少内存开销
        super().__init__(
            image_size=224,    # 使用标准尺寸
            patch_size=32,     # 使用最大patch尺寸减少序列长度
            num_layers=12,     # 使用最小层数
            num_heads=12,
            hidden_dim=768,    # 使用最小隐藏维度
            mlp_dim=3072,
            dropout=0.0,
            attention_dropout=0.0,
            num_classes=1000,
            representation_size=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_weights=False
        )
        # 删除encoder blocks,后续会重新构建
        del self.encoder.layers
        # 将所有参数转换为bool类型
        self.convert_params_to_bool()


class ViTBuilder(BaseModelBuilder):
    def __init__(self):
        self.base_models = nn.ModuleDict()
        self.layer_cache = LayerCache()
        self.init_base_model()

    def init_base_model(self):
        """初始化ViT基础模型"""
        self.base_models['standard'] = BaseViT()

    def _create_conv_proj(self, **kwargs) -> nn.Conv2d:
        """创建卷积投影层的工厂函数"""
        conv_proj = nn.Conv2d(
            in_channels=kwargs['in_channels'],
            out_channels=kwargs['out_channels'],
            kernel_size=kwargs['kernel_size'],
            stride=kwargs['kernel_size']
        )
        fan_in = conv_proj.in_channels * \
            conv_proj.kernel_size[0] * conv_proj.kernel_size[1]
        nn.init.trunc_normal_(conv_proj.weight, std=math.sqrt(1 / fan_in))
        if conv_proj.bias is not None:
            nn.init.zeros_(conv_proj.bias)
        return conv_proj

    def _create_encoder(self, **kwargs) -> Encoder:
        """创建编码器的工厂函数"""
        # 计算序列长度
        image_size = kwargs['image_size']  # 必须提供图像尺寸
        patch_size = kwargs['patch_size']
        seq_length = (image_size // patch_size) ** 2 + 1  # +1 是为了 class token

        # 创建缓存键，包含图像尺寸和patch尺寸信息
        cache_key = f'encoder_{image_size}_{patch_size}_{kwargs["hidden_dim"]}'
        
        encoder = Encoder(
            seq_length=seq_length,
            num_layers=kwargs['num_layers'],
            num_heads=kwargs['num_heads'],
            hidden_dim=kwargs['hidden_dim'],
            mlp_dim=kwargs['mlp_dim'],
            dropout=kwargs.get('dropout', 0.0),
            attention_dropout=kwargs.get('attention_dropout', 0.0),
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

        # 创建位置编码
        encoder.pos_embedding = nn.Parameter(
            torch.zeros(1, seq_length, kwargs['hidden_dim']))
        
        return encoder

    def _create_heads(self, **kwargs) -> nn.Sequential:
        """创建头部层的工厂函数"""
        hidden_dim = kwargs['hidden_dim']
        num_classes = kwargs['num_classes']
        representation_size = kwargs.get('representation_size')

        heads_layers = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(
                hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        heads = nn.Sequential(heads_layers)

        if "pre_logits" in heads_layers:
            fan_in = heads_layers["pre_logits"].in_features
            nn.init.trunc_normal_(
                heads_layers["pre_logits"].weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(heads_layers["pre_logits"].bias)

        nn.init.zeros_(heads_layers["head"].weight)
        nn.init.zeros_(heads_layers["head"].bias)

        return heads

    def make_layer(self, config: dict, **kwargs) -> nn.Module:
        """实现抽象方法 make_layer"""
        # 使用完整的配置信息创建唯一的缓存键
        cache_name = f'encoder_{config["image_size"]}_{config["patch_size"]}_{config["hidden_dim"]}'
        return self.layer_cache.get_or_create(
            cache_name=cache_name,
            creator_fn=self._create_encoder,
            **config
        )

    def get_model(self, model_name: str) -> nn.Module:
        """获取ViT模型"""
        configs = {
            'vit_b_16': {
                'image_size': 224,
                'patch_size': 16,
                'num_layers': 12,
                'num_heads': 12,
                'hidden_dim': 768,
                'mlp_dim': 3072,
                'dropout': 0.0,
                'attention_dropout': 0.0,
                'num_classes': 1000,
                'representation_size': None
            },
            'vit_b_32': {
                'image_size': 224,
                'patch_size': 32,
                'num_layers': 12,
                'num_heads': 12,
                'hidden_dim': 768,
                'mlp_dim': 3072,
                'dropout': 0.0,
                'attention_dropout': 0.0,
                'num_classes': 1000,
                'representation_size': None
            },
            'vit_l_16': {
                'image_size': 224,
                'patch_size': 16,
                'num_layers': 24,
                'num_heads': 16,
                'hidden_dim': 1024,
                'mlp_dim': 4096,
                'dropout': 0.0,
                'attention_dropout': 0.0,
                'num_classes': 1000,
                'representation_size': None
            },
            'vit_h_14': {
                'image_size': 518,  # 修改为正确的图像尺寸
                'patch_size': 14,
                'num_layers': 32,
                'num_heads': 16,
                'hidden_dim': 1280,
                'mlp_dim': 5120,
                'dropout': 0.0,
                'attention_dropout': 0.0,
                'num_classes': 1000,
                'representation_size': None
            }
        }

        if model_name not in configs:
            raise ValueError(f"Unsupported ViT model: {model_name}")

        config = configs[model_name]
        model = copy.deepcopy(self.base_models['standard'])

        # 更新模型参数
        model.image_size = config['image_size']
        model.patch_size = config['patch_size']
        model.num_layers = config['num_layers']
        model.num_heads = config['num_heads']
        model.hidden_dim = config['hidden_dim']
        model.mlp_dim = config['mlp_dim']
        model.representation_size = config['representation_size']
        model.num_classes = config['num_classes']

        # 使用通用缓存系统获取或创建各层
        model.conv_proj = self.layer_cache.get_or_create(
            cache_name='conv_proj',
            creator_fn=self._create_conv_proj,
            in_channels=3,
            out_channels=config['hidden_dim'],
            kernel_size=config['patch_size']
        )

        model.encoder = self.make_layer(config)

        model.heads = self.layer_cache.get_or_create(
            cache_name='heads',
            creator_fn=self._create_heads,
            hidden_dim=config['hidden_dim'],
            num_classes=config['num_classes'],
            representation_size=config['representation_size']
        )

        # 更新 class_token
        model.class_token = nn.Parameter(
            torch.zeros(1, 1, config['hidden_dim']))

        model.convert_params_to_bool()
        return model
