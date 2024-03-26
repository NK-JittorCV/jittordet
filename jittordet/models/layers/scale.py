# Copyright (c) OpenMMLab. All rights reserved.
import jittor as jt
import jittor.nn as nn


class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = jt.float32(scale)

    def execute(self, x):
        return x * self.scale
