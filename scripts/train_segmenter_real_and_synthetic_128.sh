#!/bin/bash

GPU_IDS="0"
NUM_GPUS=1

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} helpers/segmentor_trainer.py \
 --name segmenter_real_and_synthetic_cityscapes_128 \
 --dataset "cityscapes" --max_dim 1024 --dim 128 \
 --save_latest_freq 1 --num_workers 16 --gpu_ids ${GPU_IDS} \
 --batch_size 16 \
 --eval_freq 10  --log_freq 50 \
 --x_model deeplabv3 --niter 300 --x_pretrained_path "./datasets/resnet101-imagenet.pth" \
 --x_not_restore_last \
 --x_synthetic_dataset \
 --s_discretization max --s_which_iter 128 --s_load_path "./checkpoints/LAYOUT_SYNTHESIZER_FOLDER" \
 --s_cond_seg "semantic" --s_cond_mode "sem_assisted-entropy-spread" --s_joints_mul 1 \
 --soft_sem_seg \
 --estimated_cond --estimator_load_path "checkpoints/COND_ESTIMATOR_FOLDER" \
 --i_which_iter 199 --i_load_path "./checkpoints/IMAGE_SYNTHESIZER_FOLDER" \
 --x_duo --d_dataset "cityscapes"