#!/bin/bash

GPU_IDS="0"
NUM_GPUS=1

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} helpers/visualizer.py \
 --name face_editing_celeba_256 \
 --dataset "celeba" --max_dim 256 --dim 256 --no_v_flip --addition_mode \
 --save_latest_freq 1000 --num_workers 4 --gpu_ids ${GPU_IDS} \
 --s_cond_seg "semantic" --s_cond_mode "sem_assisted-entropy-spread-inter-ment-novelty" --s_joints_mul 1 \
 --niter 100 --s_t 0.1 \
 --batch_size 2 \
 --s_discretization 'max' --soft_sem_seg \
 --estimated_cond --estimator_load_path "checkpoints/2020-10-09-09:49:10-partial_layout_image_synthesizer_celeba_256" \
 --s_switch_cond --s_lambda_novelty 50 --s_seg_type "completor" \
 --s_which_iter 256 --s_load_path "checkpoints/2020-10-09-09:49:10-partial_layout_image_synthesizer_celeba_256" \
 --i_img_type "style_generator" \
 --i_which_iter 256 --i_load_path "checkpoints/2020-10-09-09:49:10-partial_layout_image_synthesizer_celeba_256" \
 --vis_method 'face-hair-teeth-unbald-moreborws-hat-glasses' --vis_steps 16 \
 --extraction_path "logs/STYLE_EXTRACTION_FOLDER" --save_full_res --vis_random_style