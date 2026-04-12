"""
FP8 Llama model using COAT's mixed-granularity activation quantization.

Architecture: Quartet-II's Llama (RoPE, SwiGLU, flash-attn).
FP8 ops:     COAT's Triton activation kernels + FP8CacheWeightModule.

COAT concepts implemented:
  - Coat_quantize_bgn : quantizes input to FP8 once before the first block;
                        in the backward it passes the quantized gradient onward.
  - BeforeAttentionResidual : FP8 RMSNorm + FP8 QKV projections.
  - AfterAttentionResidual  : FP8 O projection + FP8 residual add.
  - MLPResidual             : FP8 RMSNorm + FP8 gate/up/down + FP8 SiLU/mul + FP8 residual.
  - Coat_quantize_end : passes through in the forward; in the backward it
                        quantizes the incoming gradient for the first block.

Memory flow (per block):
  forward  → (re_x: BF16 residual, Qx: FP8 activation, Sx: scale)
  backward ← gradient flows in the reverse order through the same autograd fns.
"""

import math
import sys
import os

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

# ── COAT imports ──────────────────────────────────────────────────────────────
_SRC_DIR      = os.path.dirname(os.path.abspath(__file__))   # src/models/
_SRC_PARENT   = os.path.dirname(_SRC_DIR)                     # src/
_PROJECT_ROOT = os.path.dirname(_SRC_PARENT)                  # fp8-pretrain/
for _p in [_SRC_PARENT, _PROJECT_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from third_party.coat.activation.real_quantization import (
    Coat_quantize_bgn,
    Coat_quantize_end,
    fp8_add_Ifp_Ifp_Ofp_Og16,
    fp8_add_Ifp_Ifp_Ofp_Opt,
    fp8_division,
    fp8_division_transpose,
    fp8_linear_backward,
    fp8_linear_forward,
    fp8_mul_backward,
    fp8_mul_forward,
    fp8_quantize_pertensor,
    fp8_quantize_pertensor_transpose,
    fp8_rmsnorm_backward,
    fp8_rmsnorm_forward,
    fp8_silu_backward,
    fp8_silu_forward,
    fp8_transpose,
)
from third_party.coat.utils._fp8_quantization_config import QuantizationConfig
from third_party.coat.utils._fp8manager import FP8Manager
from third_party.coat.utils._fp8_weightcache import FP8CacheWeightModule

from models.base import GPTBase
from models.llama import RMSNorm, precompute_freqs_cis, apply_rotary_emb


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Before-Attention Residual  —  FP8 RMSNorm + QKV projections
# ─────────────────────────────────────────────────────────────────────────────

class FP8BeforeAttentionResidual(FP8CacheWeightModule):
    """FP8 RMSNorm followed by three FP8 linear projections (Q, K, V)."""

    def __init__(self, config, qargs: QuantizationConfig, layer_idx: int):
        # Pass None as HF config — FP8CacheWeightModule stores it but never uses it.
        super().__init__(None, qargs, layer_idx)
        self.qargs = qargs
        self.fwobits = {
            "fabit": qargs.fabit, "fwbit": qargs.fwbit, "fobit": qargs.fobit,
            "babit": qargs.babit, "bwbit": qargs.bwbit, "bobit": qargs.bobit,
        }
        n_embd, n_head = config.n_embd, config.n_head
        head_dim = n_embd // n_head
        self.q_proj = nn.Linear(n_embd, n_head * head_dim, bias=False)
        self.k_proj = nn.Linear(n_embd, n_head * head_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, n_head * head_dim, bias=False)

    def forward(self, re_x, x, s, rmsnorm_weight):
        if self.training:
            with torch.no_grad():
                w1_s = self.prepare_weight(self.q_proj.weight, "q_proj", FP8Manager.is_first_microbatch)
                w2_s = self.prepare_weight(self.k_proj.weight, "k_proj", FP8Manager.is_first_microbatch)
                w3_s = self.prepare_weight(self.v_proj.weight, "v_proj", FP8Manager.is_first_microbatch)
            # memory-efficient path: w, w_t are None; scale is pre-computed
            return _FP8BeforeAttentionResidual.apply(
                re_x, x, s,
                self.q_proj.weight, None, None, w1_s,
                self.k_proj.weight, None, None, w2_s,
                self.v_proj.weight, None, None, w3_s,
                rmsnorm_weight,
                self.qargs.group_size, self.fwobits, self.qargs,
            )
        else:
            x_norm = F.rms_norm(re_x, rmsnorm_weight.shape, rmsnorm_weight)
            return re_x, self.q_proj(x_norm), self.k_proj(x_norm), self.v_proj(x_norm)


class _FP8BeforeAttentionResidual(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        re_x, in_x, in_s,
        w1_origin, w1, w1_t, w1_s,
        w2_origin, w2, w2_t, w2_s,
        w3_origin, w3, w3_t, w3_s,
        rmsnorm_weight,
        group_size, fwobits, qargs,
        eps=1e-5,
    ):
        in_x = in_x.view(torch.float8_e4m3fn)

        # FP8 RMSNorm
        ln_x, ln_s, ln_x_t, ln_utils = fp8_rmsnorm_forward(
            in_x, in_s, rmsnorm_weight, group_size, eps, transpose_output_2d=True
        )

        # Recompute FP8 weights from BF16 origin + cached scale (memory-efficient)
        w1, w1_s = fp8_division(w1_origin, qargs.group_size, fwobits["fwbit"], w1_s)
        w2, w2_s = fp8_division(w2_origin, qargs.group_size, fwobits["fwbit"], w2_s)
        w3, w3_s = fp8_division(w3_origin, qargs.group_size, fwobits["fwbit"], w3_s)

        q = fp8_linear_forward(ln_x, ln_s, w1, w1_s, False, group_size)
        k = fp8_linear_forward(ln_x, ln_s, w2, w2_s, False, group_size)
        v = fp8_linear_forward(ln_x, ln_s, w3, w3_s, False, group_size)

        ctx.save_for_backward(in_x, in_s, ln_x_t, ln_s)
        # store BF16 origin + scale (NOT the FP8 weight) to save memory
        ctx.weight = (w1_origin, w1_s, w2_origin, w2_s, w3_origin, w3_s)
        ctx.group_size = group_size
        ctx.ln_utils = ln_utils
        ctx.fwobits = fwobits
        ctx.qargs = qargs

        return re_x, q, k, v

    @staticmethod
    def backward(ctx, fp_grad, q_g, k_g, v_g):
        in_x, in_s, ln_x_t, ln_s = ctx.saved_tensors
        w1_t, w1_s, w2_t, w2_s, w3_t, w3_s = ctx.weight
        group_size = ctx.group_size
        rms_weight, rstd, num_warps = ctx.ln_utils
        fwobits = ctx.fwobits
        qargs = ctx.qargs

        # Quantize per-tensor (attention gradients come in BF16 from SDPA)
        q_g, q_gs, q_g_t = fp8_quantize_pertensor_transpose(
            q_g, group_size, fwobits["babit"], transpose_output_2d=True, stochastic=False
        )
        k_g, k_gs, k_g_t = fp8_quantize_pertensor_transpose(
            k_g, group_size, fwobits["babit"], transpose_output_2d=True, stochastic=False
        )
        v_g, v_gs, v_g_t = fp8_quantize_pertensor_transpose(
            v_g, group_size, fwobits["babit"], transpose_output_2d=True, stochastic=False
        )

        # Recompute transposed FP8 weights for grad_input
        w1_t, w1_s = fp8_division_transpose(w1_t, qargs.group_size, fwobits["fwbit"], w1_s, only_transposed=True)
        w2_t, w2_s = fp8_division_transpose(w2_t, qargs.group_size, fwobits["fwbit"], w2_s, only_transposed=True)
        w3_t, w3_s = fp8_division_transpose(w3_t, qargs.group_size, fwobits["fwbit"], w3_s, only_transposed=True)

        fc_g1, wg1 = fp8_linear_backward(ln_x_t, ln_s, q_g, q_gs, q_g_t, w1_t, w1_s, group_size)
        fc_g2, wg2 = fp8_linear_backward(ln_x_t, ln_s, k_g, k_gs, k_g_t, w2_t, w2_s, group_size)
        fc_g3, wg3 = fp8_linear_backward(ln_x_t, ln_s, v_g, v_gs, v_g_t, w3_t, w3_s, group_size)
        fc_g = fc_g1 + fc_g2 + fc_g3

        # FP8 RMSNorm backward
        in_g, rms_wg = fp8_rmsnorm_backward(in_x, in_s, fc_g, rms_weight, rstd, group_size, num_warps)

        # Residual add + quantize for propagation to the previous block
        re_g, (in_g, in_sg, in_sg_g16) = fp8_add_Ifp_Ifp_Ofp_Opt(
            fp_grad, in_g, group_size, fwobits["babit"], stochastic=False
        )
        in_g = in_g.view(torch.float8_e4m3fn)

        return (
            re_g, in_g, in_sg_g16,
            wg1, None, None, None,   # q_proj weight grad
            wg2, None, None, None,   # k_proj weight grad
            wg3, None, None, None,   # v_proj weight grad
            rms_wg,                  # rmsnorm weight grad
            None, None, None,        # group_size, fwobits, qargs
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  After-Attention Residual  —  FP8 O projection + residual add
# ─────────────────────────────────────────────────────────────────────────────

class FP8AfterAttentionResidual(FP8CacheWeightModule):
    """FP8 output projection followed by a FP8 residual add."""

    def __init__(self, config, qargs: QuantizationConfig, layer_idx: int):
        super().__init__(None, qargs, layer_idx)
        self.qargs = qargs
        self.fwobits = {
            "fabit": qargs.fabit, "fwbit": qargs.fwbit, "fobit": qargs.fobit,
            "babit": qargs.babit, "bwbit": qargs.bwbit, "bobit": qargs.bobit,
        }
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, re_x, attn_output):
        if self.training:
            with torch.no_grad():
                w4_s = self.prepare_weight(self.o_proj.weight, "o_proj", FP8Manager.is_first_microbatch)
            return _FP8AfterAttentionResidual.apply(
                re_x, attn_output,
                self.o_proj.weight, None, None, w4_s,
                self.qargs.group_size, self.fwobits, self.qargs,
            )
        else:
            return re_x + self.o_proj(attn_output), None, None


class _FP8AfterAttentionResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, re_x, flash_x, w4_origin, w4, w4_t, w4_s, group_size, fwobits, qargs):
        # Quantize attention output per-tensor for FP8 linear
        flash_qx, flash_s, _ = fp8_quantize_pertensor(flash_x, group_size, fwobits["fabit"])

        # FP8 O projection
        w4, w4_s = fp8_division(w4_origin, qargs.group_size, fwobits["fwbit"], w4_s)
        fc4_x = fp8_linear_forward(flash_qx, flash_s, w4, w4_s, False, group_size)

        # FP8 residual add → BF16 output + FP8(group-16) for the next block
        fp_x, (out_x, out_s) = fp8_add_Ifp_Ifp_Ofp_Og16(re_x, fc4_x, flash_qx.dtype, group_size)

        ctx.save_for_backward(flash_x, flash_s)
        ctx.weight = (w4_origin, w4_s)
        ctx.group_size = group_size
        ctx.fwobits = fwobits
        ctx.qargs = qargs

        out_x = out_x.view(torch.float8_e4m3fn)
        return fp_x, out_x, out_s

    @staticmethod
    def backward(ctx, fp_grad, out_g, out_gs):
        flash_x, flash_s = ctx.saved_tensors
        w4_t, w4_s = ctx.weight
        group_size = ctx.group_size
        fwobits = ctx.fwobits
        qargs = ctx.qargs

        out_g = out_g.view(torch.float8_e5m2)
        out_gs_max = out_gs.max()

        out_g_t = fp8_transpose(out_g, transpose_output_2d=True)
        # Recompute transposed flash input (not stored to save memory)
        flash_x_t, flash_s = fp8_division_transpose(
            flash_x, group_size, fwobits["fabit"], flash_s, stochastic=False, only_transposed=True
        )
        w4_t, w4_s = fp8_division_transpose(w4_t, qargs.group_size, fwobits["fwbit"], w4_s, only_transposed=True)
        fc4_g, w4_wg = fp8_linear_backward(flash_x_t, flash_s, out_g, out_gs_max, out_g_t, w4_t, w4_s, group_size)

        return fp_grad, fc4_g, w4_wg, None, None, None, None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MLP Residual  —  FP8 RMSNorm + SwiGLU + down projection + residual
# ─────────────────────────────────────────────────────────────────────────────

class FP8MLPResidual(FP8CacheWeightModule):
    """FP8 RMSNorm + gate/up projections + FP8 SiLU + mul + down projection + residual add."""

    def __init__(self, config, qargs: QuantizationConfig, layer_idx: int):
        super().__init__(None, qargs, layer_idx)
        self.qargs = qargs
        self.fwobits = {
            "fabit": qargs.fabit, "fwbit": qargs.fwbit, "fobit": qargs.fobit,
            "babit": qargs.babit, "bwbit": qargs.bwbit, "bobit": qargs.bobit,
        }
        n_embd = config.n_embd
        hidden_dim = n_embd * 4
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        self.hidden_dim = hidden_dim

        self.gate_proj = nn.Linear(n_embd, hidden_dim, bias=False)
        self.up_proj   = nn.Linear(n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, n_embd, bias=False)

    def forward(self, re_x, x, s, rmsnorm_weight):
        if self.training:
            with torch.no_grad():
                w1_s = self.prepare_weight(self.gate_proj.weight, "gate_proj", FP8Manager.is_first_microbatch)
                w2_s = self.prepare_weight(self.up_proj.weight,   "up_proj",   FP8Manager.is_first_microbatch)
                w3_s = self.prepare_weight(self.down_proj.weight, "down_proj", FP8Manager.is_first_microbatch)
            return _FP8MLPResidual.apply(
                re_x, x, s,
                self.gate_proj.weight, None, None, w1_s,
                self.up_proj.weight,   None, None, w2_s,
                self.down_proj.weight, None, None, w3_s,
                rmsnorm_weight,
                self.qargs.group_size, self.fwobits, self.qargs,
            )
        else:
            x_norm = F.rms_norm(re_x, rmsnorm_weight.shape, rmsnorm_weight)
            gate = F.silu(self.gate_proj(x_norm))
            up   = self.up_proj(x_norm)
            return re_x + self.down_proj(gate * up), None, None


class _FP8MLPResidual(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        re_x, in_x, in_s,
        w1_origin, w1, w1_t, w1_s,
        w2_origin, w2, w2_t, w2_s,
        w3_origin, w3, w3_t, w3_s,
        rmsnorm_weight,
        group_size, fwobits, qargs,
        eps=1e-5,
    ):
        in_x = in_x.view(torch.float8_e4m3fn)

        # FP8 RMSNorm
        ln_x, ln_s, ln_x_t, ln_utils = fp8_rmsnorm_forward(
            in_x, in_s, rmsnorm_weight, group_size, eps, transpose_output_2d=True
        )

        # FP8 gate, up, down projections (weights recomputed from BF16 + scale)
        w1, w1_s = fp8_division(w1_origin, qargs.group_size, fwobits["fwbit"], w1_s)
        w2, w2_s = fp8_division(w2_origin, qargs.group_size, fwobits["fwbit"], w2_s)
        w3, w3_s = fp8_division(w3_origin, qargs.group_size, fwobits["fwbit"], w3_s)

        gate_x, gate_s = fp8_linear_forward(ln_x, ln_s, w1, w1_s, True, group_size)
        up_x,   up_s   = fp8_linear_forward(ln_x, ln_s, w2, w2_s, True, group_size)

        # FP8 SiLU activation on gate
        silu_x, silu_s = fp8_silu_forward(gate_x, gate_s, group_size)

        # FP8 element-wise mul(silu(gate), up)
        mul_x, mul_s, mul_x_t = fp8_mul_forward(
            silu_x, silu_s, up_x, up_s, group_size, transpose_output_2d=True
        )

        # FP8 down projection
        fc3_x = fp8_linear_forward(mul_x, mul_s, w3, w3_s, False, group_size)

        # FP8 residual add → BF16 + FP8(group-16) for the next block
        fp_x, (out_x, out_s) = fp8_add_Ifp_Ifp_Ofp_Og16(re_x, fc3_x, mul_x.dtype, group_size)

        ctx.save_for_backward(
            in_x, in_s, ln_x_t, ln_s,
            gate_x, gate_s, up_x, up_s,
            silu_x, silu_s, mul_x_t, mul_s,
        )
        ctx.weight = (w1_origin, w1_s, w2_origin, w2_s, w3_origin, w3_s)
        ctx.group_size = group_size
        ctx.ln_utils = ln_utils
        ctx.fwobits = fwobits
        ctx.qargs = qargs

        out_x = out_x.view(torch.float8_e4m3fn)
        return fp_x, out_x, out_s

    @staticmethod
    def backward(ctx, fp_grad, out_g, out_gs):
        (in_x, in_s, ln_x_t, ln_s,
         gate_x, gate_s, up_x, up_s,
         silu_x, silu_s, mul_x_t, mul_s) = ctx.saved_tensors
        w1_t, w1_s, w2_t, w2_s, w3_t, w3_s = ctx.weight
        group_size = ctx.group_size
        rms_weight, rstd, num_warps = ctx.ln_utils
        fwobits = ctx.fwobits
        qargs = ctx.qargs

        out_g = out_g.view(torch.float8_e5m2)
        out_gs_max = out_gs.max()

        # Down projection backward
        out_g_t = fp8_transpose(out_g, transpose_output_2d=True)
        w3_t, w3_s = fp8_division_transpose(w3_t, qargs.group_size, fwobits["fwbit"], w3_s, only_transposed=True)
        fc3_g, wg3 = fp8_linear_backward(mul_x_t, mul_s, out_g, out_gs_max, out_g_t, w3_t, w3_s, group_size)
        del out_g, out_g_t, w3_t

        # Element-wise mul backward
        mul_g1, (mul_g2, mul_gs2, mul_g2_t) = fp8_mul_backward(
            silu_x, silu_s, up_x, up_s, fc3_g, group_size, fwobits["babit"],
            output_quantized_transpose=True,
        )

        # SiLU backward
        silu_g, silu_gs, silu_g_t = fp8_silu_backward(
            gate_x, gate_s, mul_g1, group_size, fwobits["babit"],
            output_quantized_transpose=True,
        )

        # Gate and up projection backward
        w1_t, w1_s = fp8_division_transpose(w1_t, qargs.group_size, fwobits["fwbit"], w1_s, only_transposed=True)
        w2_t, w2_s = fp8_division_transpose(w2_t, qargs.group_size, fwobits["fwbit"], w2_s, only_transposed=True)
        fc1_g, wg1 = fp8_linear_backward(ln_x_t, ln_s, silu_g, silu_gs, silu_g_t, w1_t, w1_s, group_size)
        fc2_g, wg2 = fp8_linear_backward(ln_x_t, ln_s, mul_g2, mul_gs2, mul_g2_t, w2_t, w2_s, group_size)
        fc_g = fc1_g + fc2_g

        # FP8 RMSNorm backward
        in_g, rms_wg = fp8_rmsnorm_backward(in_x, in_s, fc_g, rms_weight, rstd, group_size, num_warps)

        # Residual add + quantize for propagation
        re_g, (in_g, in_sg, in_sg_g16) = fp8_add_Ifp_Ifp_Ofp_Opt(
            fp_grad, in_g, group_size, fwobits["babit"], stochastic=False
        )
        in_g = in_g.view(torch.float8_e4m3fn)

        return (
            re_g, in_g, in_sg_g16,
            wg1, None, None, None,   # gate_proj weight grad
            wg2, None, None, None,   # up_proj weight grad
            wg3, None, None, None,   # down_proj weight grad
            rms_wg,                  # rmsnorm weight grad
            None, None, None,        # group_size, fwobits, qargs
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FP8 Transformer Block
# ─────────────────────────────────────────────────────────────────────────────

class FP8LlamaBlock(nn.Module):
    """Full FP8 transformer block: RoPE attention + SwiGLU MLP, all in FP8."""

    def __init__(self, config, qargs: QuantizationConfig, layer_idx: int):
        super().__init__()
        self.n_head  = config.n_head
        self.n_embd  = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.qkv_clipping = getattr(config, 'qkv_clipping', False)
        self.qkv_clipping_factor = getattr(config, 'qkv_clipping_factor', 1.0)

        # RMSNorm weights — accessed via .weight by the residual modules
        self.ln_1 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)

        # FP8 residual sub-modules (hold the actual linear weight tensors)
        self.before_attn = FP8BeforeAttentionResidual(config, qargs, layer_idx)
        self.after_attn  = FP8AfterAttentionResidual(config, qargs, layer_idx)
        self.mlp         = FP8MLPResidual(config, qargs, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,       # BF16 residual  (B, T, C)
        quant_hidden:  torch.Tensor,       # FP8 activation (B, T, C)
        scale_hidden:  torch.Tensor,       # scale          (B, T, C//g)
        freqs_cis:     torch.Tensor,
    ):
        B, T, C = hidden_states.shape

        # ── Attention ────────────────────────────────────────────────────────
        # FP8 RMSNorm + QKV projections
        residual, q, k, v = self.before_attn(
            hidden_states, quant_hidden, scale_hidden, self.ln_1.weight
        )

        if self.qkv_clipping:
            q = q.clamp(min=-self.qkv_clipping_factor, max=self.qkv_clipping_factor)
            k = k.clamp(min=-self.qkv_clipping_factor, max=self.qkv_clipping_factor)
            v = v.clamp(min=-self.qkv_clipping_factor, max=self.qkv_clipping_factor)

        # Apply RoPE (Quartet-II style: stack-based, compilable)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        q = q.transpose(1, 2)  # (B, nh, T, hs)
        k = k.transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # SDPA / Flash Attention (BF16 — only projections are FP8)
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)

        # FP8 O projection + residual add
        hidden_states, quant_hidden, scale_hidden = self.after_attn(residual, attn_out)

        # ── MLP ──────────────────────────────────────────────────────────────
        hidden_states, quant_hidden, scale_hidden = self.mlp(
            hidden_states, quant_hidden, scale_hidden, self.ln_2.weight
        )

        return hidden_states, quant_hidden, scale_hidden


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Full FP8 Llama Model
# ─────────────────────────────────────────────────────────────────────────────

class FP8Llama(GPTBase):
    """
    Llama model with COAT FP8 training (activation + optimizer quantization).

    Architecture mirrors Quartet-II's Llama (RoPE, SwiGLU, no positional embedding)
    but with FP8 layers throughout every transformer block.

    Usage:
        model = FP8Llama(args, qargs)
        opt   = CoatAdamW(model.parameters(), lr=..., qargs=qargs)
    """

    def __init__(self, config, qargs: QuantizationConfig):
        # Bypass GPTBase.__init__ (which builds BF16 blocks) and call nn.Module directly.
        nn.Module.__init__(self)
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config
        self.qargs  = qargs
        self.tokenizer = tiktoken.get_encoding("gpt2")

        self.head_dim  = config.n_embd // config.n_head
        self.freqs_cis = precompute_freqs_cis(self.head_dim, config.sequence_length)

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([
                FP8LlamaBlock(config, qargs, i) for i in range(config.n_layer)
            ]),
            ln_f = RMSNorm(config.n_embd, eps=config.rmsnorm_eps),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # FP8 quantization boundary modules
        self.quantize_input  = Coat_quantize_bgn(qargs)
        self.quantize_output = Coat_quantize_end(qargs)

        # Weight initialisation
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("o_proj.weight") or pn.endswith("down_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=self.config.init_std / math.sqrt(2 * config.n_layer))

    # ── GPTBase interface ────────────────────────────────────────────────────

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(self, idx, targets=None, get_logits=False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.sequence_length, (
            f"Cannot forward sequence of length {t}, "
            f"block size is only {self.config.sequence_length}"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Token embedding → dropout
        x = self.transformer.drop(self.transformer.wte(idx))

        # Pre-computed RoPE frequencies for this sequence length
        freqs_cis = self.freqs_cis.to(x.device)[pos]

        # Quantize input to FP8 once before all blocks.
        # Training: returns (x_bf16, Qx_fp8, Sx_scale)
        # Eval:     returns (x_bf16, None, None)
        x, Qx, Sx = self.quantize_input(x)

        for block in self.transformer.h:
            x, Qx, Sx = block(x, Qx, Sx, freqs_cis)

        # Mark end of FP8 region (no-op in forward; quantizes gradient in backward).
        x = self.quantize_output(x, Qx, Sx)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return {"logits": logits if get_logits else None, "loss": loss}
