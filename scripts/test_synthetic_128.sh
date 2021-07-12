#!/bin/bash

GPU_IDS="0"
NUM_GPUS=1

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} helpers/tester.py \
 --name test_synthetic_cityscapes_128 \
 --dataset "cityscapes" --max_dim 1024 --dim 128 \
 --x_model deeplabv3 --x_which_iter 299 --x_load_path "./checkpoints/SEGMENTER_TRAINED_ON_REAL_FOLDER" \
 --num_workers 4 --gpu_ids ${GPU_IDS} \
 --batch_size 16 \
 --niter 5 --d_dataset "cityscapes" --d_estimated_cond --d_estimator_load_path "checkpoints/COND_ESTIMATOR_FOLDER" \
 --x_synthetic_dataset \
 --s_discretization max --s_which_iter 128 --s_load_path "checkpoints/LAYOUT_SYNTHESIZER_FOLDER" \
 --s_cond_seg "semantic" --s_cond_mode "sem_assisted-entropy-spread" --s_joints_mul 1 \
 --soft_sem_seg \
 --estimated_cond --estimator_load_path "checkpoints/COND_ESTIMATOR_FOLDER" \
 --i_which_iter 199 --i_load_path "./checkpoints/IMAGE_SYNTHESIZER_FOLDER" \
 --x_which_iter_2 299 --x_load_path_2 "./checkpoints/SEGMENTER_TRAINED_ON_SYNTHETIC_FOLDER"