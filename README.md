# fp8-pretrain

## Directory layout

```
fp8-pretrain/
├── README.md
├── requirements.txt
├── setup_env.sh               # create conda env + compile COAT CUDA kernels
├── scripts/
│   └── train.sh               # single/multi-GPU launch (H100)
├── src/
│   ├── main.py
│   ├── config/base.py         # all CLI args (includes FP8 flags)
│   ├── data/                  # C4, SlimPajama, OpenWebText2, … loaders
│   ├── models/
│   │   ├── base.py            # BF16 GPT base
│   │   ├── llama.py           # BF16 Llama baseline
│   │   ├── fp8_llama.py       # ← Llama with COAT FP8 ops (main new file)
│   │   └── utils.py
│   ├── optim/                 # training loop, LR schedules, weight averaging
│   ├── logger/                # WandB + dynamics logger
│   └── distributed/           # DDP backend abstraction
└── third_party/
    └── coat/                  # self-contained COAT copy (no internet needed)
        ├── activation/real_quantization/   # Triton FP8 kernels
        ├── optimizer/fp8_adamw.py          # CoatAdamW
        └── utils/                         # QuantizationConfig, FP8Manager, …
```

## Setup

```bash
pip install --upgrade pip setuptools

export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130

https://mjunya.com/flash-attention-prebuild-wheels/
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu130torch2.9-cp311-cp311-linux_x86_64.whl

python -r requirements.txt

в third_party/coat/optimizer/kernels
pip install --no-build-isolation -e .
```

## Credits

- [COAT: Compressing Optimizer states and Activation for Memory-Efficient FP8 Training](https://arxiv.org/abs/2410.19313) — Dettmers et al., 2024
- Quartet-II — training infrastructure baseline
