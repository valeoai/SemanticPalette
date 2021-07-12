#!/bin/bash

GPU_IDS="0"
NUM_GPUS=1

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} helpers/visualizer.py \
 --name demo_latent_interpoaltion_idd_128 \
 --dataset "idd" --max_dim 1024 --dim 128  \
 --save_latest_freq 1000 --num_workers 4 --gpu_ids ${GPU_IDS} \
 --s_cond_seg "semantic" --s_cond_mode "sem_assisted-entropy-spread-inter" --s_joints_mul 1 \
 --niter 40 --s_t 0.1 \
 --batch_size 16 \
 --s_discretization 'max' --s_store_masks --soft_sem_seg \
 --estimated_cond --estimator_load_path "checkpoints/2020-10-16-09:57:04-layout_image_synthesizer_idd_128" \
 --s_which_iter 128 --s_load_path "checkpoints/2020-10-16-09:57:04-layout_image_synthesizer_idd_128" \
 --i_which_iter 128 --i_load_path "checkpoints/2020-10-16-09:57:04-layout_image_synthesizer_idd_128" \
 --vis_method 'latent' --vis_steps 16 --save_full_res