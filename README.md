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
│   ├── optim/
│   │   ├── __init__.py
│   │   ├── base.py                 # base optimizer utilities
│   │   ├── optimization.py         # training loop / optimization logic
│   │   ├── utils.py                # optimizer helper utilities
│   │   ├── weight_averaging.py     # EMA / SWA / parameter averaging
│   │   ├── memory_efficient/       # memory-efficient optimizers
│   │   │   ├── __init__.py
│   │   │   ├── apollo/
│   │   │   ├── fira/
│   │   │   ├── frugal/
│   │   │   ├── galore/
│   │   │   └── ldadam/
│   │   └── sota_opt/               # state-of-the-art optimizers
│   │       ├── __init__.py
│   │       ├── Adan/
│   │       ├── MARS/
│   │       ├── dion/
│   │       └── soap/
│   │       └── swan/
│   ├── logger/                # WandB + dynamics logger
│   └── distributed/           # DDP backend abstraction
└── third_party/
    └── coat/                  # self-contained COAT copy
        ├── activation/real_quantization/   # Triton FP8 kernels
        ├── optimizer/fp8_adamw.py          # CoatAdamW
        └── utils/                         # QuantizationConfig, FP8Manager, …
```

## Setup

I tested this on 8xH200 and 8xH100 pods (with a minor differences in commands due to the older cuda version on the latter)

```bash
pip install --upgrade pip setuptools

# specify path to cude in case of system not seeing nvcc
# e.g.
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

# we assume you've python 3.13 and CUDA 13
pip install -r requirements.txt

# COAT:
# inside third_party/coat/optimizer/kernels
pip install --no-build-isolation -e .
```

### Alternative path

Download appropriate for your cuda torch and flash attn. Other steps are the same

```bash
# install torch
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130

# use https://mjunya.com/flash-attention-prebuild-wheels/ to get an appropriate version of flash-attention package

pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu130torch2.9-cp311-cp311-linux_x86_64.whl

# exclude these both from requirements.txt
pip install -r requirements.txt
```


## Credits

- [Quartet-II codebase](https://github.com/IST-DASLab/Quartet-II/) — training infrastructure baseline
- [COAT: Compressing Optimizer states and Activation for Memory-Efficient FP8 Training](https://arxiv.org/abs/2410.19313) — method itself and training infrastructure baselinee
