#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python ./src/main.py \
    --model llama \
    --dataset_dir "./../../../datasets/" \
    --dataset fineweb \
    --opt "d-muon" \
    --lr 3e-3 \
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
    --scheduler cos \
    --warmup_steps 2000 \
    --dropout 0 \
    --beta1 0.8 --beta2 0.999 \
    --eval_interval 115 \
    --latest_ckpt_interval 1000 \
    --log_interval 4 \
    --wandb
