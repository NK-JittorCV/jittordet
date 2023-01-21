# Modified from OpenMMLab mmdet/structures/det_data_sample.py
# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

from .base_data_element import BaseDataElement
from .instance_data import InstanceData


class DetDataSample(BaseDataElement):

    @property
    def proposals(self) -> InstanceData:
        return self._proposals

    @proposals.setter
    def proposals(self, value: InstanceData):
        self.set_field(value, '_proposals', dtype=InstanceData)

    @proposals.deleter
    def proposals(self):
        del self._proposals

    @property
    def gt_instances(self) -> InstanceData:
        return self._gt_instances

    @gt_instances.setter
    def gt_instances(self, value: InstanceData):
        self.set_field(value, '_gt_instances', dtype=InstanceData)

    @gt_instances.deleter
    def gt_instances(self):
        del self._gt_instances

    @property
    def pred_instances(self) -> InstanceData:
        return self._pred_instances

    @pred_instances.setter
    def pred_instances(self, value: InstanceData):
        self.set_field(value, '_pred_instances', dtype=InstanceData)

    @pred_instances.deleter
    def pred_instances(self):
        del self._pred_instances

    @property
    def ignored_instances(self) -> InstanceData:
        return self._ignored_instances

    @ignored_instances.setter
    def ignored_instances(self, value: InstanceData):
        self.set_field(value, '_ignored_instances', dtype=InstanceData)

    @ignored_instances.deleter
    def ignored_instances(self):
        del self._ignored_instances


SampleList = List[DetDataSample]
OptSampleList = Optional[SampleList]
