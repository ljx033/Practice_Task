"""手写 LoRA：低秩注入 + 合并。不依赖 peft。

约定（与 eval/run.py 对齐）：
    inject_lora(model, target_modules, r=8, alpha=16) -> model   # 就地注入
    merge_lora(model) -> model                                  # 合并回原权重并移除 LoRA 分支
分支形状：A: (in_features, r)   B: (r, out_features)
前向：    y = W x + scaling * B(A x),  scaling = alpha / r
"""
from __future__ import annotations

import math
from typing import Iterable, List

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """在原始 nn.Linear 旁挂 A、B 两个低秩参数；forward 叠加 scaling * B(A x)。"""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        assert isinstance(base, nn.Linear)
        assert r > 0
        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r

        in_features = base.in_features
        out_features = base.out_features

        # A: (in, r) 用 kaiming；B: (r, out) 用零初始化 -> 一开始 ΔW = 0
        self.lora_A = nn.Parameter(torch.empty(in_features, self.r))
        self.lora_B = nn.Parameter(torch.zeros(self.r, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # 冻结原始权重
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        # 注意：A 是 (in, r)，B 是 (r, out)
        lora = (x @ self.lora_A) @ self.lora_B
        return out + lora * self.scaling


def _match_target(name: str, targets: Iterable[str]) -> bool:
    """既支持短名匹配（q_proj），也支持全名匹配（model.layers.0.self_attn.q_proj）。"""
    short = name.rsplit(".", 1)[-1]
    return short in targets or name in targets


def inject_lora(model: nn.Module, target_modules, r: int = 8, alpha: int = 16) -> nn.Module:
    """就地注入 LoRA。所有原始参数 requires_grad=False，只有 A/B 可训。"""
    # 1) 先把所有参数冻结
    for p in model.parameters():
        p.requires_grad = False

    target_set = set(target_modules) if target_modules else set()

    # 2) 收集目标 nn.Linear 的全名
    to_replace: List[str] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _match_target(name, target_set):
            to_replace.append(name)

    if not to_replace:
        raise ValueError(f"inject_lora: 未匹配到任何目标层 {target_modules}")

    # 3) 逐个替换
    for full_name in to_replace:
        if "." in full_name:
            parent_name, child_name = full_name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            child_name = full_name
        base = getattr(parent, child_name)
        setattr(parent, child_name, LoRALinear(base, r=r, alpha=alpha))

    return model


def _merge_one(lora_layer: LoRALinear) -> nn.Linear:
    base = lora_layer.base
    with torch.no_grad():
        # delta: (in, out)
        delta = (lora_layer.lora_A @ lora_layer.lora_B) * lora_layer.scaling
        base.weight.data = base.weight.data + delta.t().contiguous()
    for p in base.parameters():
        p.requires_grad = True
    return base


def merge_lora(model: nn.Module) -> nn.Module:
    """把 LoRA 权重合并回原 Linear，并移除 LoRA 分支。合并前后 forward 数值一致。"""

    def _walk(parent: nn.Module):
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                setattr(parent, name, _merge_one(child))
            else:
                _walk(child)

    _walk(model)
    return model


def lora_state_dict(model: nn.Module) -> dict:
    """只导出 LoRA 相关参数（A/B），用于保存 ckpt。"""
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if ".lora_A" in k or ".lora_B" in k
    }


def load_lora_state_dict(model: nn.Module, sd: dict) -> None:
    """把 LoRA 参数写回已注入 LoRA 的模型（严格只覆盖 lora_A/B）。"""
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    own = model.state_dict()
    to_load = {k: v for k, v in sd.items() if k in own}
    missing, unexpected = model.load_state_dict(to_load, strict=False)
    # 只允许 missing 是「非 LoRA」的参数（base 权重本来就存在）
    unexpected_lora = [k for k in unexpected if ".lora_" in k]
    if unexpected_lora:
        raise RuntimeError(f"load_lora_state_dict: unexpected LoRA keys {unexpected_lora[:3]}")