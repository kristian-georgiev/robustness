#!/bin/bash

#SBATCH -o log-4n-mean-%j
#SBATCH -c 40
#SBATCH --gres=gpu:volta:2


python -m robustness.main --data "/home/gridsan/groups/datasets/ImageNet/" \
                          --dataset custom_imagenet \
                          --epochs 100 \
                          --step-lr 30 \
                          --out-dir models_breeds/ \
                          --adv-train 0 \
                          --adv-eval 0 \
                          --arch resnet18 \
                          --workers 40 \
                          --make_circ \
                          --bicubic \
                          --num_rots 4
