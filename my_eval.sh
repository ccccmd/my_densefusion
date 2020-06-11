#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 eval.py --dataset_root ./Linemod_preprocessed\
  --model trained_models/linemod/pose_model_9_0.012599086272355984.pth\
  --refine_model trained_models/linemod/pose_refine_model_127_0.007160789261744246.pth