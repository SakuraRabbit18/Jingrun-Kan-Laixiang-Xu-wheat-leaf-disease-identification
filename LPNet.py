from functools import partial
import math
from typing import Any, Callable, List, Optional, Sequence
from torch import Tensor
import torch
from torch import nn

from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation as SElayer
from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _make_divisible, _ovewrite_named_param, handle_legacy_interface


__all__ = [
    "MobileNetV3",
    "MobileNet_V3_Large_Weights",
    "MobileNet_V3_Small_Weights",
    "mobilenet_v3_large",
    "mobilenet_v3_small",
    "get_LPNet",
]


class InvertedResidualConfig:
    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
        dilation: int,
        width_mult: float,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)





class LGCA(nn.Module):
    def __init__(self, channel, groups=4):
        super().__init__()
        k_size = 5 if channel >= 288 else 3
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            in_channels=groups,
            out_channels=groups,
            kernel_size=k_size,
            padding=(k_size-1)//2,
            groups=groups,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()
        self.groups = groups

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, self.groups, c // self.groups)
        y = self.conv(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y




class InvertedResidual(nn.Module):
    def __init__(
        self,
        cnf: InvertedResidualConfig,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = partial(SElayer, scale_activation=nn.Hardsigmoid),
    ):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )
        
        stride = 1 if cnf.dilation > 1 else cnf.stride
       
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )
        

        if cnf.use_se:
            layers.append(LGCA(cnf.expanded_channels))


        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


class ConvInceptionPool(nn.Module):
    def __init__(self, in_channels=96, out_channels=576, branch_ratio=0.25, pool_type=nn.MaxPool2d):
        super().__init__()
        total_pool_channels = int(in_channels * 3 * branch_ratio)
        id_channels = in_channels - total_pool_channels
        
        self.conv_id = nn.Conv2d(id_channels, int(out_channels * (id_channels/in_channels)), kernel_size=1)
        self.conv_hw = nn.Conv2d(int(in_channels * branch_ratio), int(out_channels * branch_ratio), kernel_size=1)
        self.conv_w = nn.Conv2d(int(in_channels * branch_ratio), int(out_channels * branch_ratio), kernel_size=1)
        self.conv_h = nn.Conv2d(int(in_channels * branch_ratio), int(out_channels * branch_ratio), kernel_size=1)
        
        self.pool_hw = pool_type(kernel_size=3, stride=1, padding=1)
        self.pool_w = pool_type(kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.pool_h = pool_type(kernel_size=(7, 1), stride=1, padding=(3, 0))
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        
        self.act = nn.Hardswish()

    def forward(self, x):
        branch_channels = int(x.size(1) * 0.25)
        id_channels = x.size(1) - 3 * branch_channels
        
        x_id, x_hw, x_w, x_h = torch.split(x, [id_channels, branch_channels, branch_channels, branch_channels], dim=1)
        
        x_id = self.conv_id(x_id)
        x_hw = self.conv_hw(x_hw)
        x_w = self.conv_w(x_w)
        x_h = self.conv_h(x_h)
        
        x_hw = self.pool_hw(x_hw)
        x_w = self.pool_w(x_w)
        x_h = self.pool_h(x_h)
        
        x_id = self.adaptive_pool(x_id)
        x_hw = self.adaptive_pool(x_hw)
        x_w = self.adaptive_pool(x_w)
        x_h = self.adaptive_pool(x_h)
        
        x = torch.cat([x_id, x_hw, x_w, x_h], dim=1)
        
        return self.act(x)


class MobileNetV3(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: List[InvertedResidualConfig],
        last_channel: int,
        num_classes: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        firstconv_output_channels = inverted_residual_setting[0].input_channels

        layers.append(
             Conv2dNormActivation(
                 3,
                 firstconv_output_channels,
                 kernel_size=3,
                 stride=2,
                 norm_layer=norm_layer,
                 activation_layer=nn.Hardswish,
             )
         )

        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        last_residual_channels = inverted_residual_setting[-1].out_channels
        
        self.features = nn.Sequential(*layers)
        self.pool = ConvInceptionPool(
            in_channels=last_residual_channels,
            out_channels=last_channel,
            branch_ratio=0.25,
            pool_type=nn.MaxPool2d
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def _mobilenet_v3_conf(
    arch: str, width_mult: float = 1.0, reduced_tail: bool = False, dilated: bool = False, **kwargs: Any
):
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    if arch == "mobilenet_v3_large":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
            bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
            bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation),
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1280 // reduce_divider)
    elif arch == "mobilenet_v3_small":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1024 // reduce_divider)
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel


def _mobilenet_v3(
    inverted_residual_setting: List[InvertedResidualConfig],
    last_channel: int,
    weights: Optional[WeightsEnum],
    progress: bool,** kwargs: Any,
) -> MobileNetV3:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = MobileNetV3(inverted_residual_setting, last_channel, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


_COMMON_META = {
    "min_size": (1, 1),
    "categories": _IMAGENET_CATEGORIES,
}


class MobileNet_V3_Large_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 5483032,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#mobilenetv3-large--small",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 74.042,
                    "acc@5": 91.340,
                }
            },
            "_ops": 0.217,
            "_file_size": 21.114,
            "_docs": """These weights were trained from scratch by using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 5483032,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-reg-tuning",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 75.274,
                    "acc@5": 92.566,
                }
            },
            "_ops": 0.217,
            "_file_size": 21.107,
            "_docs": """
                These weights improve marginally upon the results of the original paper by using a modified version of
                TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


class MobileNet_V3_Small_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 2542856,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#mobilenetv3-large--small",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 67.668,
                    "acc@5": 87.402,
                }
            },
            "_ops": 0.057,
            "_file_size": 9.829,
            "_docs": """
                These weights improve upon the results of the original paper by using a simple training recipe.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


@handle_legacy_interface(weights=("pretrained", MobileNet_V3_Large_Weights.IMAGENET1K_V1))
def my_mobilenet_v3_large(
    *, weights: Optional[MobileNet_V3_Large_Weights] = None, progress: bool = True, **kwargs: Any
) -> MobileNetV3:
    weights = MobileNet_V3_Large_Weights.verify(weights)

    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_large",** kwargs)
    return _mobilenet_v3(inverted_residual_setting, last_channel, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", MobileNet_V3_Small_Weights.IMAGENET1K_V1))
def my_mobilenet_v3_small(
    *, weights: Optional[MobileNet_V3_Small_Weights] = None, progress: bool = True, **kwargs: Any
) -> MobileNetV3:
    weights = MobileNet_V3_Small_Weights.verify(weights)

    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_small",** kwargs)
    return _mobilenet_v3(inverted_residual_setting, last_channel, weights, progress, **kwargs)

def get_LPNet(pretrained=True):
    model = my_mobilenet_v3_small(pretrained=False)
    if pretrained:
        ckpt = torch.load("./mobilenet_v3_small-047dcff4.pth")
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in ckpt.items() if k in model_dict and (v.shape == model_dict[k].shape)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
    return model

if __name__ == "__main__":
    model = get_LPNet()
    print(model)
