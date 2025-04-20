from functools import partial
from typing import Any, cast, Dict, List, Optional, Union

import torch
import torch.nn as nn
import time
import os
import csv
from datetime import datetime

from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface


__all__ = [
    "VGG",
    "VGG11_Weights",
    "VGG11_BN_Weights",
    "VGG13_Weights",
    "VGG13_BN_Weights",
    "VGG16_Weights",
    "VGG16_BN_Weights",
    "VGG19_Weights",
    "VGG19_BN_Weights",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
]


class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5,
        model_name: str = "vgg"
    ) -> None:
        super().__init__()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timing_data = []
        _log_api_usage_once(self)

        # Features timing
        t0 = time.perf_counter()
        self.features = features
        timing_data.append([timestamp, model_name, "features", (time.perf_counter() - t0) * 1000])

        # Avgpool timing
        t0 = time.perf_counter()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        timing_data.append([timestamp, model_name, "avgpool", (time.perf_counter() - t0) * 1000])

        # Classifier layers timing
        classifier_layers = []

        # FC1 layer
        t0 = time.perf_counter()
        fc1 = nn.Linear(512 * 7 * 7, 4096)
        timing_data.append([timestamp, model_name, "fc1", (time.perf_counter() - t0) * 1000])
        classifier_layers.append(fc1)

        # ReLU1
        t0 = time.perf_counter()
        relu1 = nn.ReLU(True)
        timing_data.append([timestamp, model_name, "relu1", (time.perf_counter() - t0) * 1000])
        classifier_layers.append(relu1)

        # Dropout1
        t0 = time.perf_counter()
        dropout1 = nn.Dropout(p=dropout)
        timing_data.append([timestamp, model_name, "dropout1", (time.perf_counter() - t0) * 1000])
        classifier_layers.append(dropout1)

        # FC2 layer
        t0 = time.perf_counter()
        fc2 = nn.Linear(4096, 4096)
        timing_data.append([timestamp, model_name, "fc2", (time.perf_counter() - t0) * 1000])
        classifier_layers.append(fc2)

        # ReLU2
        t0 = time.perf_counter()
        relu2 = nn.ReLU(True)
        timing_data.append([timestamp, model_name, "relu2", (time.perf_counter() - t0) * 1000])
        classifier_layers.append(relu2)

        # Dropout2
        t0 = time.perf_counter()
        dropout2 = nn.Dropout(p=dropout)
        timing_data.append([timestamp, model_name, "dropout2", (time.perf_counter() - t0) * 1000])
        classifier_layers.append(dropout2)

        # FC3 layer
        t0 = time.perf_counter()
        fc3 = nn.Linear(4096, num_classes)
        timing_data.append([timestamp, model_name, "fc3", (time.perf_counter() - t0) * 1000])
        classifier_layers.append(fc3)

        self.classifier = nn.Sequential(*classifier_layers)

        t0 = time.perf_counter()
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
        timing_data.append([timestamp, model_name, "weight_init", (time.perf_counter() - t0) * 1000])

        # Write timing data
        filename = "layer_init_timing.csv"
        file_exists = os.path.isfile(filename)
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "model_name", "component", "time_ms"])
            writer.writerows(timing_data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False, model_name: str = "vgg") -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    timing_data = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for v in cfg:
        if v == "M":
            t0 = time.perf_counter()
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            timing_data.append([timestamp, model_name, f"maxpool2d_{v}", (time.perf_counter() - t0) * 1000])
        else:
            v = cast(int, v)
            # Conv2d timing
            t0 = time.perf_counter()
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            timing_data.append([timestamp, model_name, f"conv2d_{v}", (time.perf_counter() - t0) * 1000])

            if batch_norm:
                # BatchNorm timing
                t0 = time.perf_counter()
                bn = nn.BatchNorm2d(v)
                timing_data.append([timestamp, model_name, f"bn2d_{v}", (time.perf_counter() - t0) * 1000])

                # ReLU timing
                t0 = time.perf_counter()
                relu = nn.ReLU(inplace=True)
                timing_data.append([timestamp, model_name, f"relu2d_{v}", (time.perf_counter() - t0) * 1000])

                layers += [conv2d, bn, relu]
            else:
                # ReLU timing
                t0 = time.perf_counter()
                relu = nn.ReLU(inplace=True)
                timing_data.append([timestamp, model_name, f"relu2d_{v}", (time.perf_counter() - t0) * 1000])

                layers += [conv2d, relu]
            in_channels = v

    # Write timing data
    filename = "layer_init_timing.csv"
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "model_name", "component", "time_ms"])
        writer.writerows(timing_data)

    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(cfg: str, batch_norm: bool, weights: Optional[WeightsEnum], progress: bool, **kwargs: Any) -> VGG:
    if weights is not None:
        # kwargs["init_weights"] = False
        if weights.meta["categories"] is not None:
            _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    # 根据配置确定模型名称
    model_names = {
        "A": "vgg11",
        "B": "vgg13",
        "D": "vgg16",
        "E": "vgg19"
    }
    model_name = model_names[cfg]
    if batch_norm:
        model_name += "_bn"

    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, model_name=model_name),
                model_name=model_name, **kwargs)

    if weights is not None:
        timing_data = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        t0 = time.perf_counter()
        weight = weights.get_state_dict(progress=progress, check_hash=True)
        get_time = (time.perf_counter() - t0) * 1000
        timing_data.append([timestamp, model_name, "weights_get", get_time])

        t0 = time.perf_counter()
        model.load_state_dict(weight)
        load_time = (time.perf_counter() - t0) * 1000
        timing_data.append([timestamp, model_name, "weights_load", load_time])

        filename = "layer_weight_loading.csv"
        file_exists = os.path.isfile(filename)
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "model_name", "stage", "time_ms"])
            writer.writerows(timing_data)
    return model


_COMMON_META = {
    "min_size": (32, 32),
    "categories": _IMAGENET_CATEGORIES,
    "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg",
    "_docs": """These weights were trained from scratch by using a simplified training recipe.""",
}


class VGG11_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vgg11-8a719046.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 132863336,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 69.020,
                    "acc@5": 88.628,
                }
            },
            "_ops": 7.609,
            "_file_size": 506.84,
        },
    )
    DEFAULT = IMAGENET1K_V1


class VGG11_BN_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 132868840,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 70.370,
                    "acc@5": 89.810,
                }
            },
            "_ops": 7.609,
            "_file_size": 506.881,
        },
    )
    DEFAULT = IMAGENET1K_V1


class VGG13_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vgg13-19584684.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 133047848,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 69.928,
                    "acc@5": 89.246,
                }
            },
            "_ops": 11.308,
            "_file_size": 507.545,
        },
    )
    DEFAULT = IMAGENET1K_V1


class VGG13_BN_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 133053736,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 71.586,
                    "acc@5": 90.374,
                }
            },
            "_ops": 11.308,
            "_file_size": 507.59,
        },
    )
    DEFAULT = IMAGENET1K_V1


class VGG16_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vgg16-397923af.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 138357544,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 71.592,
                    "acc@5": 90.382,
                }
            },
            "_ops": 15.47,
            "_file_size": 527.796,
        },
    )
    IMAGENET1K_FEATURES = Weights(
        # Weights ported from https://github.com/amdegroot/ssd.pytorch/
        url="https://download.pytorch.org/models/vgg16_features-amdegroot-88682ab5.pth",
        transforms=partial(
            ImageClassification,
            crop_size=224,
            mean=(0.48235, 0.45882, 0.40784),
            std=(1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0),
        ),
        meta={
            **_COMMON_META,
            "num_params": 138357544,
            "categories": None,
            "recipe": "https://github.com/amdegroot/ssd.pytorch#training-ssd",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": float("nan"),
                    "acc@5": float("nan"),
                }
            },
            "_ops": 15.47,
            "_file_size": 527.802,
            "_docs": """
                These weights can't be used for classification because they are missing values in the `classifier`
                module. Only the `features` module has valid values and can be used for feature extraction. The weights
                were trained using the original input standardization method as described in the paper.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


class VGG16_BN_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 138365992,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 73.360,
                    "acc@5": 91.516,
                }
            },
            "_ops": 15.47,
            "_file_size": 527.866,
        },
    )
    DEFAULT = IMAGENET1K_V1


class VGG19_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 143667240,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 72.376,
                    "acc@5": 90.876,
                }
            },
            "_ops": 19.632,
            "_file_size": 548.051,
        },
    )
    DEFAULT = IMAGENET1K_V1


class VGG19_BN_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 143678248,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 74.218,
                    "acc@5": 91.842,
                }
            },
            "_ops": 19.632,
            "_file_size": 548.143,
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_model()
@handle_legacy_interface(weights=("pretrained", VGG11_Weights.IMAGENET1K_V1))
def vgg11(*, weights: Optional[VGG11_Weights] = None, progress: bool = True, **kwargs: Any) -> VGG:
    """VGG-11 from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        weights (:class:`~torchvision.models.VGG11_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG11_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.VGG11_Weights
        :members:
    """
    weights = VGG11_Weights.verify(weights)

    return _vgg("A", False, weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=("pretrained", VGG11_BN_Weights.IMAGENET1K_V1))
def vgg11_bn(*, weights: Optional[VGG11_BN_Weights] = None, progress: bool = True, **kwargs: Any) -> VGG:
    """VGG-11-BN from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        weights (:class:`~torchvision.models.VGG11_BN_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG11_BN_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.VGG11_BN_Weights
        :members:
    """
    weights = VGG11_BN_Weights.verify(weights)

    return _vgg("A", True, weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=("pretrained", VGG13_Weights.IMAGENET1K_V1))
def vgg13(*, weights: Optional[VGG13_Weights] = None, progress: bool = True, **kwargs: Any) -> VGG:
    """VGG-13 from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        weights (:class:`~torchvision.models.VGG13_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG13_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.VGG13_Weights
        :members:
    """
    weights = VGG13_Weights.verify(weights)

    return _vgg("B", False, weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=("pretrained", VGG13_BN_Weights.IMAGENET1K_V1))
def vgg13_bn(*, weights: Optional[VGG13_BN_Weights] = None, progress: bool = True, **kwargs: Any) -> VGG:
    """VGG-13-BN from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        weights (:class:`~torchvision.models.VGG13_BN_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG13_BN_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.VGG13_BN_Weights
        :members:
    """
    weights = VGG13_BN_Weights.verify(weights)

    return _vgg("B", True, weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=("pretrained", VGG16_Weights.IMAGENET1K_V1))
def vgg16(*, weights: Optional[VGG16_Weights] = None, progress: bool = True, **kwargs: Any) -> VGG:
    """VGG-16 from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        weights (:class:`~torchvision.models.VGG16_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG16_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.VGG16_Weights
        :members:
    """
    weights = VGG16_Weights.verify(weights)

    return _vgg("D", False, weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=("pretrained", VGG16_BN_Weights.IMAGENET1K_V1))
def vgg16_bn(*, weights: Optional[VGG16_BN_Weights] = None, progress: bool = True, **kwargs: Any) -> VGG:
    """VGG-16-BN from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        weights (:class:`~torchvision.models.VGG16_BN_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG16_BN_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.VGG16_BN_Weights
        :members:
    """
    weights = VGG16_BN_Weights.verify(weights)

    return _vgg("D", True, weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=("pretrained", VGG19_Weights.IMAGENET1K_V1))
def vgg19(*, weights: Optional[VGG19_Weights] = None, progress: bool = True, **kwargs: Any) -> VGG:
    """VGG-19 from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        weights (:class:`~torchvision.models.VGG19_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG19_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.VGG19_Weights
        :members:
    """
    weights = VGG19_Weights.verify(weights)

    return _vgg("E", False, weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=("pretrained", VGG19_BN_Weights.IMAGENET1K_V1))
def vgg19_bn(*, weights: Optional[VGG19_BN_Weights] = None, progress: bool = True, **kwargs: Any) -> VGG:
    """VGG-19_BN from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        weights (:class:`~torchvision.models.VGG19_BN_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG19_BN_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.VGG19_BN_Weights
        :members:
    """
    weights = VGG19_BN_Weights.verify(weights)

    return _vgg("E", True, weights, progress, **kwargs)
