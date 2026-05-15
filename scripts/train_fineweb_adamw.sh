#!/bin/bash

# Probably args for distributed training, have no idea, how to handle them
# export CUDA_VISIBLE_DEVICES=1,2
# torchrun --nproc_per_node=2 --master_port=1233 ./src/main.py \
# --do_not_auto_resume --distributed_backend nccl \

export CUDA_VISIBLE_DEVICES=0
python ./src/main.py \
    --model llama \
    --dataset_dir "./../../../datasets/" \
    --dataset fineweb \
    --opt "adamw-spectral-l1-reg" \
    --lr 1e-3 \
    --iterations 16000 \
    --n_embd 768 \
    --n_head 12 \
    --n_layer 12 \
    --batch_size 64 \
    --sequence_length 512 \
    --acc_steps 4 \
    --grad_clip 0.5 \
    --seed 0 \
    --weight_decay 0.1 \
    --spectral_l1_reg_coef 0.1 \
    --scheduler cos \
    --warmup_steps 2000 \
    --dropout 0 \
    --beta1 0.9 --beta2 0.95 \
    --eval_interval 115 \
    --latest_ckpt_interval 1000 \
    --log_interval 4 \
    --wandb
