import numpy as np

from jittordet.engine import TRANSFORMS
from jittordet.structures import DetDataSample, InstanceData


@TRANSFORMS.register_module()
class PackDetInputs:
    """Pack the inputs data for detection."""
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
    }

    def __init__(self,
                 meta_keys=('sample_idx', 'img_id', 'img_path', 'ori_shape',
                            'img_shape', 'scale_factor', 'flip',
                            'flip_direction')):
        self.meta_keys = meta_keys

    def __call__(self, results):
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            packed_results['inputs'] = img

        if 'gt_ignore_flags' in results:
            valid_idx = np.where(results['gt_ignore_flags'] == 0)[0]
            ignore_idx = np.where(results['gt_ignore_flags'] == 1)[0]

        data_sample = DetDataSample()
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()

        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if 'gt_ignore_flags' in results:
                instance_data[
                    self.mapping_table[key]] = results[key][valid_idx]
                ignore_instance_data[
                    self.mapping_table[key]] = results[key][ignore_idx]
            else:
                instance_data[self.mapping_table[key]] = results[key]
        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data

        img_meta = {}
        for key in self.meta_keys:
            assert key in results, f'`{key}` is not found in `results`, ' \
                f'the valid keys are {list(results)}.'
            img_meta[key] = results[key]

        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
