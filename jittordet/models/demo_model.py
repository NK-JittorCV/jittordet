import jittor.nn as nn

from jittordet.engine import MODELS


@MODELS.register_module()
class DemoModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(256, 256, 3)

    def execute(self, data, phase='loss'):
        data = data[0]
        loss = self.conv(data).mean()
        if phase == 'loss':
            return dict(loss=loss)
        elif phase == 'predict':
            return [loss]
