#!/bin/bash

GPU_IDS="0"
NUM_GPUS=1

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} helpers/visualizer.py \
 --name demo_partial_editing_cityscapes_128 \
 --dataset "cityscapes" --max_dim 1024 --dim 128 \
 --num_workers 4 --gpu_ids ${GPU_IDS} \
 --s_cond_seg "semantic" --s_cond_mode "sem_assisted-entropy-spread-inter-ment" --s_joints_mul 1 \
 --niter 30 \
 --batch_size 16 --vis_dataloader_bs 16 \
 --s_discretization 'none' --soft_sem_seg --s_vertical_sem_crop --s_seg_type "completor" \
 --estimated_cond --estimator_load_path "checkpoints/2020-10-22-21:12:51-partial_layout_image_synthesizer_cityscapes_128" \
 --s_which_iter 128 --s_load_path "checkpoints/2020-10-22-21:12:51-partial_layout_image_synthesizer_cityscapes_128" \
 --i_which_iter 128 --i_load_path "checkpoints/2020-10-22-21:12:51-partial_layout_image_synthesizer_cityscapes_128" \
 --vis_method 'scrop' --vis_steps 16 --save_full_res \