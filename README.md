# JittorDet

## introduction

JittorDet is an object detection benchmark based on [Jittor](https://cg.cs.tsinghua.edu.cn/jittor/).

## Supported Models

JittorDet supports commonly used datasets (COCO, VOC) and models (RetinaNet, Faster R-CNN) out of box.

Currently supported models are as below:

- RetinaNet
- Faster R-CNN
- GFocalLoss

New state-of-the-art models are being implemented.

## Getting Started

### Install

Please first follow the [tutorial](https://github.com/Jittor/jittor) to install jittor.
Here, we recommend using jittor==1.3.6.10, which we have tested on.

Then, install the `jittordet` by running:
```
pip install -v -e .
```

If you want to use multi-gpu training or testing, please install OpenMPI
```
sudo apt install openmpi-bin openmpi-common libopenmpi-dev
```

### Training

We support single-gpu, multi-gpu training.
```
#Single-GPU
python tools/train.py {CONFIG_PATH}

# Multi-GPU
bash tools/dist_train.sh {CONFIG_PATH} {NUM_GPUS}
```

### Testing

We support single-gpu, multi-gpu testing.
```
#Single-GPU
python tools/test.py {CONFIG_PATH}

# Multi-GPU
bash tools/dist_test.sh {CONFIG_PATH} {NUM_GPUS}
```
