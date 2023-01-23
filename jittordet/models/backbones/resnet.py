# Modified from jdet/python/models/backbones/resnet.py
import jittor as jt
from jittor import nn

from jittordet.engine import MODELS


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    conv = nn.Conv(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation)
    jt.init.relu_invariant_gauss_(conv.weight, mode='fan_out')
    return conv


def conv1x1(in_planes, out_planes, stride=1):
    conv = nn.Conv(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    jt.init.relu_invariant_gauss_(conv.weight, mode='fan_out')
    return conv


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if (norm_layer is None):
            norm_layer = nn.BatchNorm
        if ((groups != 1) or (base_width != 64)):
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if (dilation > 1):
            raise NotImplementedError(
                'Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.Relu()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if (self.downsample is not None):
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(Bottleneck, self).__init__()
        if (norm_layer is None):
            norm_layer = nn.BatchNorm
        width = (int((planes * (base_width / 64.0))) * groups)
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, (planes * self.expansion))
        self.bn3 = norm_layer((planes * self.expansion))
        self.relu = nn.Relu()
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if (self.downsample is not None):
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


@MODELS.register_module()
class ResNet(nn.Module):

    structures = {
        18: {
            'layers': [2, 2, 2, 2],
            'block': BasicBlock
        },
        34: {
            'layers': [3, 4, 6, 3],
            'block': BasicBlock
        },
        50: {
            'layers': [3, 4, 6, 3],
            'block': Bottleneck
        },
        101: {
            'layers': [3, 4, 23, 3],
            'block': Bottleneck
        },
        152: {
            'layers': [3, 8, 36, 3],
            'block': Bottleneck
        },
    }

    def __init__(self,
                 depth,
                 return_stages=['layer4'],
                 frozen_stages=-1,
                 norm_eval=True,
                 num_classes=None,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 pretrained=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if (norm_layer is None):
            norm_layer = nn.BatchNorm
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if (replace_stride_with_dilation is None):
            replace_stride_with_dilation = [False, False, False]
        if (len(replace_stride_with_dilation) != 3):
            raise ValueError(
                'replace_stride_with_dilation should be None or a 3-element '
                f'tuple, but got {replace_stride_with_dilation}')
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        jt.init.relu_invariant_gauss_(self.conv1.weight, mode='fan_out')
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.Relu()
        self.maxpool = nn.Pool(
            kernel_size=3, stride=2, padding=1, op='maximum')

        assert depth in self.structures
        structure = self.structures[depth]
        block, layers = structure['block'], structure['layers']
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2])
        self.num_classes = num_classes
        self.return_stages = return_stages
        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear((512 * block.expansion), num_classes)
        self.pretrained = pretrained
        self._freeze_stages()

    def init_weights(self):
        assert self.pretrained is not None
        self.load(self.pretrained)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if ((stride != 1) or (self.inplanes != (planes * block.expansion))):
            downsample = nn.Sequential(
                conv1x1(self.inplanes, (planes * block.expansion), stride),
                norm_layer((planes * block.expansion)))
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, norm_layer))
        self.inplanes = (planes * block.expansion)
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.stop_grad()

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.stop_grad()

    def execute(self, x):
        outputs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for i in range(1, 5):
            name = f'layer{i}'
            x = getattr(self, name)(x)
            if name in self.return_stages:
                outputs.append(x)
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = jt.reshape(x, (x.shape[0], -1))
            x = self.fc(x)
            if 'fc' in self.return_stages:
                outputs.append(x)
        return tuple(outputs)

    def train(self):
        super(ResNet, self).train()
        self._freeze_stages()
        if self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm):
                    m.eval()


@MODELS.register_module()
class ResNetV1d(nn.Module):

    structures = {
        18: {
            'layers': [2, 2, 2, 2],
            'block': BasicBlock
        },
        34: {
            'layers': [3, 4, 6, 3],
            'block': BasicBlock
        },
        50: {
            'layers': [3, 4, 6, 3],
            'block': Bottleneck
        },
        101: {
            'layers': [3, 4, 23, 3],
            'block': Bottleneck
        },
        152: {
            'layers': [3, 8, 36, 3],
            'block': Bottleneck
        },
    }

    def __init__(self,
                 depth,
                 return_stages=['layer4'],
                 frozen_stages=-1,
                 norm_eval=True,
                 num_classes=None,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None,
                 pretrained=None):
        super(ResNetV1d, self).__init__()
        if (norm_layer is None):
            norm_layer = nn.BatchNorm
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if (replace_stride_with_dilation is None):
            replace_stride_with_dilation = [False, False, False]
        if (len(replace_stride_with_dilation) != 3):
            raise ValueError(
                'replace_stride_with_dilation should be None or a 3-element '
                f'tuple, got {replace_stride_with_dilation}')
        self.groups = groups
        self.base_width = width_per_group
        self.C1 = nn.Sequential(
            nn.Conv(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(32),
            nn.Relu(),
            nn.Conv(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(32),
            nn.Relu(),
            nn.Conv(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(64),
            nn.Relu(),
        )

        assert depth in self.structures
        structure = self.structures[depth]
        block, layers = structure['block'], structure['layers']

        self.relu = nn.Relu()
        self.maxpool = nn.Pool(
            kernel_size=3, stride=2, padding=1, op='maximum')
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2])
        self.num_classes = num_classes
        self.return_stages = return_stages
        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear((512 * block.expansion), num_classes)

        self.pretrained = pretrained
        self._freeze_stages()

    def init_weights(self):
        assert self.pretrained is not None
        self.load(self.pretrained)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if ((stride != 1) or (self.inplanes != (planes * block.expansion))):
            downsample = nn.Sequential(
                nn.Pool(stride, stride=stride, op='mean'),
                conv1x1(self.inplanes, (planes * block.expansion), 1),
                norm_layer((planes * block.expansion)))
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, norm_layer))
        self.inplanes = (planes * block.expansion)
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def execute(self, x):
        outputs = []
        x = self.C1(x)
        x = self.maxpool(x)
        for i in range(1, 5):
            name = f'layer{i}'
            x = getattr(self, name)(x)
            if name in self.return_stages:
                outputs.append(x)
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = jt.reshape(x, (x.shape[0], -1))
            x = self.fc(x)
            if 'fc' in self.return_stages:
                outputs.append(x)
        return outputs

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.stop_grad()

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.stop_grad()

    def train(self):
        super(ResNetV1d, self).train()
        self._freeze_stages()
        if self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm):
                    m.eval()
