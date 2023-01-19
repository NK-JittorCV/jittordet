#!/usr/bin/env bash

CONFIG=$1
GPUS=$2

mpirun -np $GPUS python tools/train.py \
    $CONFIG
    ${@:3}
