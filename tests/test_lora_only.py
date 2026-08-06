import torch.nn as nn

from optim.memory_efficient.hybrid_lora import (
    HybridLoRALinear,
    apply_hybrid_lora,
    freeze_non_lora_parameters,
    get_hybrid_lora_param_groups,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 16)
        self.attn = nn.ModuleDict({"q_proj": nn.Linear(16, 16, bias=False)})
        self.mlp = nn.ModuleDict({"up_proj": nn.Linear(16, 48, bias=False)})
        self.norm = nn.LayerNorm(16)
        self.lm_head = nn.Linear(16, 32, bias=False)


def test_only_lora_adapters_are_trainable():
    model = TinyModel()
    apply_hybrid_lora(model, rank=4, alpha=4, scope="all", verbose=False)
    freeze_non_lora_parameters(model)

    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable_names
    assert all(name.endswith(("lora_A", "lora_B")) for name in trainable_names)
    assert isinstance(model.attn["q_proj"], HybridLoRALinear)
    assert isinstance(model.mlp["up_proj"], HybridLoRALinear)
    assert model.attn["q_proj"].rank == 4
    assert model.mlp["up_proj"].rank == 4

    base_groups, lora_groups = get_hybrid_lora_param_groups(
        model,
        weight_decay=0.1,
        base_lr=1e-3,
        lora_lr=1e-3,
    )
    assert base_groups == []
    assert lora_groups
