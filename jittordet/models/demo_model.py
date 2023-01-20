import jittor.nn as nn

from jittordet.engine import MODELS


@MODELS.register_module()
class DemoModel(nn.Module):

    def __init__(self, preprocessor):
        super().__init__()
        self.preprocessor = MODELS.build(preprocessor)
        self.conv = nn.Conv2d(3, 1, 3)

    def execute(self, data, phase='loss'):
        data = self.preprocessor(data, training=True)
        import pdb
        pdb.set_trace()
        results = self.conv(data['inputs'])
        return dict(loss=results.sum())
