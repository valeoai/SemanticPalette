#!/bin/bash

python helpers/cond_estimator_trainer.py \
 --name palette_estimator_cityscapes \
 --dataset "cityscapes" --max_dim 512 \
 --batch_size 16 --num_workers 4 \
 --estimator_min_components 1 --estimator_max_components 30