# modified from mmdetection.datasets.xml_style
import os.path as osp
import xml.etree.ElementTree as ET

from PIL import Image

from ..engine import DATASETS
from .base import BaseDetDataset


@DATASETS.register_module()
class VocDataset(BaseDetDataset):

    METAINFO = {
        'classes':
        ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
         'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
         'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
        # PALETTE is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157), (0, 182, 199),
                    (0, 226, 252), (182, 182, 255), (0, 0, 230), (220, 20, 60),
                    (163, 255, 0), (0, 82, 0), (3, 95, 161), (0, 80, 100),
                    (183, 130, 88)]
    }

    def load_data_list(self):
        assert self.metainfo.get('classes', None) is not None, \
            'CLASSES in `VocDataset` can not be None.'
        self.cat2label = {
            cat: i
            for i, cat in enumerate(self.metainfo['classes'])
        }

        data_list = []
        for img_id in open(self.ann_file, 'r'):
            img_id = img_id.strip()
            img_path = osp.join(self.img_path, f'{img_id}.jpg')
            xml_path = osp.join(self.xml_path, f'{img_id}.xml')

            raw_img_info = {}
            raw_img_info['img_id'] = img_id
            raw_img_info['img_path'] = img_path
            raw_img_info['xml_path'] = xml_path

            parsed_data_info = self.parse_data_info(raw_img_info)
            data_list.append(parsed_data_info)
        return data_list

    def parse_data_info(self, img_info):
        data_info = {}
        data_info['img_id'] = img_info['img_id']
        data_info['img_path'] = img_info['img_path']
        data_info['xml_path'] = img_info['xml_path']

        # deal with xml file
        raw_ann_info = ET.parse(data_info['xml_path'])
        root = raw_ann_info.getroot()
        size = root.find('size')
        if size is not None:
            width = int(size.find('width').text)
            height = int(size.find('height').text)
        else:
            img = Image.open(img_info['img_path'])
            height, width = img.height, img.width
            del img

        data_info['height'] = height
        data_info['width'] = width

        instances = []
        for obj in raw_ann_info.findall('object'):
            instance = {}
            name = obj.find('name').text
            if name not in self._metainfo['classes']:
                continue
            difficult = obj.find('difficult')
            difficult = 0 if difficult is None else int(difficult.text)
            bnd_box = obj.find('bndbox')
            bbox = [
                int(float(bnd_box.find('xmin').text)) - 1,
                int(float(bnd_box.find('ymin').text)) - 1,
                int(float(bnd_box.find('xmax').text)) - 1,
                int(float(bnd_box.find('ymax').text)) - 1
            ]
            instance['ignore_flag'] = difficult
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[name]
            instances.append(instance)
        data_info['instances'] = instances
        return data_info

    def filter_data(self):
        if self.test_mode:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False) \
            if self.filter_cfg is not None else False
        min_size = self.filter_cfg.get('min_size', 0) \
            if self.filter_cfg is not None else 0

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and len(data_info['instances']) == 0:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos
