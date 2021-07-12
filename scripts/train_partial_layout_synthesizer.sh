#!/bin/bash

GPU_IDS="0"
NUM_GPUS=1

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} helpers/seg_generator_trainer.py \
 --name partial_layout_synthesizer_cityscapes \
 --dataset "cityscapes" --max_dim 1024 \
 --save_latest_freq 1000 --num_workers 4 --gpu_ids ${GPU_IDS} \
 --s_cond_seg "semantic" --s_cond_mode "sem_assisted-entropy-spread-inter-ment" --s_joints_mul 1 \
 --niter 150 --s_batch_size_per_res 1024 512 256 64 16 6 4 2 1 --s_t 0.1 \
 --s_iter_function_per_res cycle cycle cycle cycle iter iter iter iter iter \
 --s_step_mul_per_res 4 3 3 2 2 1 1 1 1 \
 --s_save_at_every_res --s_discretization 'none' --soft_sem_seg \
 --estimated_cond --estimator_load_path "checkpoints/COND_ESTIMATOR_FOLDER" \
 --s_seg_type "completor" --s_vertical_sem_crop