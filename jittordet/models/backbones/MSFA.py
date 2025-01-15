# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import jittor as jt
from jittor import nn
from jittordet.engine import MODELS
from kymatio.torch import Scattering2D

@MODELS.register_module()
class MSFA(nn.Module):
    def __init__(self, backbone, use_sar=True, use_wavelet=False, input_size=(800,800)):
        self.use_sar = use_sar
        self.use_wavelet = use_wavelet
        self.input_size=input_size
        if use_sar and not use_wavelet :
            self.in_channels = 3 
        elif use_sar and use_wavelet:
            self.in_channels = 1
        elif not use_sar:
            self.in_channels = 0
        if use_wavelet:
            self.in_channels += 81
            self.wavelet_trans = Scattering2D(J=2, shape=self.input_size)
        backbone['in_channels'] = self.in_channels
        self.backbone = MODELS.build(backbone)
    def execute(self, x):
        xs = []
        if self.use_sar and not self.use_wavelet:
            return self.backbone(x)
        x_ = x.mean(1,keepdim=True)
        with jt.no_grad():
            if self.use_sar and self.use_wavelet:
                xs.append(x_)
            if self.use_wavelet:
                out = nn.functional.interpolate(self.wavelet_trans(x_).squeeze(1), self.input_size, mode='bilinear')
                xs.append(out)
            x = jt.cat(xs,1)
        x = self.backbone(x)
        return x


    def init_weights(self):
        super(MSFA, self).init_weights()


