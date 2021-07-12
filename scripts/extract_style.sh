#!/bin/bash

GPU_IDS="0"
NUM_GPUS=1

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} helpers/style_extractor.py \
 --name extract_style_celeba_256 \
 --dataset "celeba" --max_dim 256 --batch_size 16 \
 --niter 200 \
 --num_workers 4 --gpu_ids ${GPU_IDS} \
 --i_img_type "style_generator" \
 --i_which_iter 256 --i_load_path "checkpoints/2020-10-09-09:49:10-partial_layout_image_synthesizer_celeba_256" \



