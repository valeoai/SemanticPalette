#!/bin/bash

GPU_IDS="0"
NUM_GPUS=1

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} helpers/img_generator_trainer.py \
 --name image_synthesizer_cityscapes_128 \
 --dataset "cityscapes" --max_dim 128 --batch_size 8 --log_freq 100 \
 --save_latest_freq 1 --num_workers 16 --gpu_ids ${GPU_IDS} \
 --niter 100 --niter_decay 100 \



