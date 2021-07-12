#!/bin/bash

GPU_IDS="0"
NUM_GPUS=1

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} helpers/segmentor_trainer.py \
 --name segmenter_real_cityscapes_128 \
 --dataset "cityscapes" --max_dim 128 \
 --save_latest_freq 1 --num_workers 16 --gpu_ids ${GPU_IDS} \
 --batch_size 16 \
 --eval_freq 10  --log_freq 50 \
 --x_model deeplabv3 --niter 300 --x_pretrained_path "./datasets/resnet101-imagenet.pth" \
 --x_not_restore_last