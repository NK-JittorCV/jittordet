#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3

mpirun -np $GPUS python tools/test.py \
    $CONFIG
    $CHECKPOINT
    ${@:4}
