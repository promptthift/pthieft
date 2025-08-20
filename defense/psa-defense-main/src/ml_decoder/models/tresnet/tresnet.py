import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn import Module as Module
from collections import OrderedDict

from .layers.anti_aliasing import AntiAliasDownsampleLayer
from .layers.avg_pool import FastAvgPool2d
from .layers.general_layers import SEModule, SpaceToDepthModule
# from inplace_abn import InPlaceABN, ABN

from typing import Optional

class ABN(nn.Module):
    """Activated Batch Normalization

    This gathers a BatchNorm and an activation function in a single module

    Args:
        num_features: Number of feature channels in the input and output
        eps: Small constant to prevent numerical issues
        momentum: Momentum factor applied to compute running statistics with
            exponential moving average, or `None` to compute running statistics
            with cumulative moving average
        affine: If `True` apply learned scale and shift transformation after normalization
        track_running_stats: a boolean value that when set to `True`, this
            module tracks the running mean and variance, and when set to `False`,
            this module does not track such statistics and uses batch statistics instead
            in both training and eval modes if the running mean and variance are `None`
        activation: Name of the activation functions, one of: `relu`, `leaky_relu`,
            `elu` or `identity`
        activation_param: Negative slope for the `leaky_relu` activation or `alpha`
            parameter for the `elu` activation
    """

    _version = 2
    __constants__ = [
        "track_running_stats",
        "momentum",
        "eps",
        "num_features",
        "affine",
        "activation",
        "activation_param",
    ]
    num_features: int
    eps: float
    momentum: Optional[float]
    affine: bool
    track_running_stats: bool
    activation: str
    activation_param: float

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        activation: str = "leaky_relu",
        activation_param: float = 0.01,
    ):
        super(ABN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.activation = activation
        self.activation_param = activation_param
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
            self.register_parameter("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def _get_momentum_and_training(self):
        if self.momentum is None:
            momentum = 0.0
        else:
            momentum = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    momentum = 1.0 / float(self.num_batches_tracked)
                else:
                    momentum = self.momentum

        if self.training:
            training = True
        else:
            training = (self.running_mean is None) and (self.running_var is None)

        return momentum, training

    def _get_running_stats(self):
        running_mean = (
            self.running_mean if not self.training or self.track_running_stats else None
        )
        running_var = (
            self.running_var if not self.training or self.track_running_stats else None
        )
        return running_mean, running_var

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        momentum, training = self._get_momentum_and_training()
        running_mean, running_var = self._get_running_stats()

        x = functional.batch_norm(
            x,
            running_mean,
            running_var,
            self.weight,
            self.bias,
            training,
            momentum,
            self.eps,
        )

        if self.activation == "relu":
            return functional.relu(x, inplace=True)
        elif self.activation == "leaky_relu":
            return functional.leaky_relu(
                x, negative_slope=self.activation_param, inplace=True
            )
        elif self.activation == "elu":
            return functional.elu(x, alpha=self.activation_param, inplace=True)
        elif self.activation == "identity":
            return x
        else:
            raise RuntimeError(f"Unknown activation function {self.activation}")

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + "num_batches_tracked"
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(ABN, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def extra_repr(self):
        rep = "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, activation={activation}"
        if self.activation in ["leaky_relu", "elu"]:
            rep += "[{activation_param}]"
        return rep.format(**self.__dict__)

'''
def InplacABN_to_ABN(module: nn.Module) -> nn.Module:
    # convert all InplaceABN layer to bit-accurate ABN layers.
    if isinstance(module, InPlaceABN):
        module_new = ABN(module.num_features, activation=module.activation,
                         activation_param=module.activation_param)
        for key in module.state_dict():
            module_new.state_dict()[key].copy_(module.state_dict()[key])
        module_new.training = module.training
        module_new.weight.data = module_new.weight.abs() + module_new.eps
        return module_new
    for name, child in reversed(module._modules.items()):
        new_child = InplacABN_to_ABN(child)
        if new_child != child:
            module._modules[name] = new_child
    return module
'''

def conv2d(ni, nf, stride):
    return nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(nf),
        nn.ReLU(inplace=True)
    )


def conv2d_ABN(ni, nf, stride, activation="leaky_relu", kernel_size=3, activation_param=1e-2, groups=1):
    return nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=groups,
                  bias=False),
        # InPlaceABN(num_features=nf, activation=activation, activation_param=activation_param)
        ABN(num_features=nf, activation=activation, activation_param=activation_param)
    )


class BasicBlock(Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None):
        super(BasicBlock, self).__init__()
        if stride == 1:
            self.conv1 = conv2d_ABN(inplanes, planes, stride=1, activation_param=1e-3)
        else:
            if anti_alias_layer is None:
                self.conv1 = conv2d_ABN(inplanes, planes, stride=2, activation_param=1e-3)
            else:
                self.conv1 = nn.Sequential(conv2d_ABN(inplanes, planes, stride=1, activation_param=1e-3),
                                           anti_alias_layer(channels=planes, filt_size=3, stride=2))

        self.conv2 = conv2d_ABN(planes, planes, stride=1, activation="identity")
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        reduce_layer_planes = max(planes * self.expansion // 4, 64)
        self.se = SEModule(planes * self.expansion, reduce_layer_planes) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.se is not None: out = self.se(out)

        out += residual

        out = self.relu(out)

        return out


class Bottleneck(Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv2d_ABN(inplanes, planes, kernel_size=1, stride=1, activation="leaky_relu",
                                activation_param=1e-3)
        if stride == 1:
            self.conv2 = conv2d_ABN(planes, planes, kernel_size=3, stride=1, activation="leaky_relu",
                                    activation_param=1e-3)
        else:
            if anti_alias_layer is None:
                self.conv2 = conv2d_ABN(planes, planes, kernel_size=3, stride=2, activation="leaky_relu",
                                        activation_param=1e-3)
            else:
                self.conv2 = nn.Sequential(conv2d_ABN(planes, planes, kernel_size=3, stride=1,
                                                      activation="leaky_relu", activation_param=1e-3),
                                           anti_alias_layer(channels=planes, filt_size=3, stride=2))

        self.conv3 = conv2d_ABN(planes, planes * self.expansion, kernel_size=1, stride=1,
                                activation="identity")

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        reduce_layer_planes = max(planes * self.expansion // 8, 64)
        self.se = SEModule(planes, reduce_layer_planes) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None: out = self.se(out)

        out = self.conv3(out)
        out = out + residual  # no inplace
        out = self.relu(out)

        return out


class TResNet(Module):

    def __init__(self, layers, in_chans=3, num_classes=1000, width_factor=1.0, first_two_layers=BasicBlock):
        super(TResNet, self).__init__()

        # JIT layers
        space_to_depth = SpaceToDepthModule()
        anti_alias_layer = AntiAliasDownsampleLayer
        global_pool_layer = FastAvgPool2d(flatten=True)

        # TResnet stages
        self.inplanes = int(64 * width_factor)
        self.planes = int(64 * width_factor)
        conv1 = conv2d_ABN(in_chans * 16, self.planes, stride=1, kernel_size=3)
        layer1 = self._make_layer(first_two_layers, self.planes, layers[0], stride=1, use_se=True,
                                  anti_alias_layer=anti_alias_layer)  # 56x56
        layer2 = self._make_layer(first_two_layers, self.planes * 2, layers[1], stride=2, use_se=True,
                                  anti_alias_layer=anti_alias_layer)  # 28x28
        layer3 = self._make_layer(Bottleneck, self.planes * 4, layers[2], stride=2, use_se=True,
                                  anti_alias_layer=anti_alias_layer)  # 14x14
        layer4 = self._make_layer(Bottleneck, self.planes * 8, layers[3], stride=2, use_se=False,
                                  anti_alias_layer=anti_alias_layer)  # 7x7

        # body
        self.body = nn.Sequential(OrderedDict([
            ('SpaceToDepth', space_to_depth),
            ('conv1', conv1),
            ('layer1', layer1),
            ('layer2', layer2),
            ('layer3', layer3),
            ('layer4', layer4)]))

        # head
        self.global_pool = nn.Sequential(OrderedDict([('global_pool_layer', global_pool_layer)]))
        self.num_features = (self.planes * 8) * Bottleneck.expansion
        fc = nn.Linear(self.num_features, num_classes)
        self.head = nn.Sequential(OrderedDict([('fc', fc)]))

        # model initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, ABN): # isinstance(m, InPlaceABN):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # residual connections special initialization
        for m in self.modules():
            if isinstance(m, BasicBlock):
                m.conv2[1].weight = nn.Parameter(torch.zeros_like(m.conv2[1].weight))  # BN to zero
            if isinstance(m, Bottleneck):
                m.conv3[1].weight = nn.Parameter(torch.zeros_like(m.conv3[1].weight))  # BN to zero
            if isinstance(m, nn.Linear): m.weight.data.normal_(0, 0.01)

    def _make_layer(self, block, planes, blocks, stride=1, use_se=True, anti_alias_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = []
            if stride == 2:
                # avg pooling before 1x1 conv
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False))
            layers += [conv2d_ABN(self.inplanes, planes * block.expansion, kernel_size=1, stride=1,
                                  activation="identity")]
            downsample = nn.Sequential(*layers)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=use_se,
                            anti_alias_layer=anti_alias_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks): layers.append(
            block(self.inplanes, planes, use_se=use_se, anti_alias_layer=anti_alias_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.body(x)
        self.embeddings = self.global_pool(x)
        logits = self.head(self.embeddings)
        return logits


def TResnetS(model_params):
    """Constructs a small TResnet model.
    """
    in_chans = 3
    num_classes = model_params['num_classes']
    args = model_params['args']
    model = TResNet(layers=[3, 4, 6, 3], num_classes=num_classes, in_chans=in_chans)
    return model

def TResnetM(model_params):
    """Constructs a medium TResnet model.
    """
    in_chans = 3
    num_classes = model_params['num_classes']
    model = TResNet(layers=[3, 4, 11, 3], num_classes=num_classes, in_chans=in_chans)
    return model


def TResnetL(model_params):
    """Constructs a large TResnet model.
    """
    in_chans = 3
    num_classes = model_params['num_classes']
    layers_list = [3, 4, 23, 3]
    model = TResNet(layers=layers_list, num_classes=num_classes, in_chans=in_chans, first_two_layers=Bottleneck)
    return model

def TResnetXL(model_params):
    """Constructs a large TResnet model.
    """
    in_chans = 3
    num_classes = model_params['num_classes']
    layers_list = [3, 8, 34, 5]
    model = TResNet(layers=layers_list, num_classes=num_classes, in_chans=in_chans, first_two_layers=Bottleneck)
    return model