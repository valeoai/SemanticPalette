#!/bin/bash

GPU_IDS="0"
NUM_GPUS=1

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} helpers/seg_img_generator_trainer.py \
 --name layout_image_synthesizer_cityscapes_128 \
 --dataset "cityscapes" --max_dim 1024 --dim 128 --batch_size 4 \
 --save_freq 25 --save_latest_freq 1 --num_workers 4 --gpu_ids ${GPU_IDS} \
 --niter 200 --eval_freq 10  --log_freq 50 \
 --s_which_iter 128 --s_load_path "./checkpoints/LAYOUT_SYNTHESIZER_FOLDER" \
 --s_cond_seg "semantic" --s_cond_mode "sem_assisted-entropy-spread-inter" --s_joints_mul 1 \
 --s_discretization 'none' --soft_sem_seg \
 --estimated_cond --estimator_load_path "checkpoints/COND_ESTIMATOR_FOLDER" \
 --s_lr 0.0001 \
 --i_which_iter 199 --i_load_path "./checkpoints/IMAGE_SYNTHESIZER_FOLDER" \
 --i_use_d2 --fake_from_real_dis "d" --fake_from_fake_dis "d2" \
 --i_lambda_feat 10 --i_lambda_vgg 10 --i_lambda_d2 0.01