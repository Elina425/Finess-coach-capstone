"""
Low-rank personalization for xLSTM exercise models (LoRA / DoRA-style).

Population-trained encoders misread idiosyncratic but valid kinematics; we keep the
base weights frozen and learn a small user-specific residual on selected linear maps.

- **LoRA** (Hu et al.): W x + (alpha/r) * B A x on chosen layers.
- **DoRA-style** (Liu et al., simplified here): same low-rank path plus a learnable
  per-output magnitude scale on the adaptation branch (no full weight decomposition).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Set, Union

import torch
import torch.nn as nn

PersonalizationMethod = Literal["lora", "dora", "bottleneck"]


class LoRALinear(nn.Module):
    """Frozen inner Linear + trainable low-rank delta (LoRA)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = max(1, int(rank))
        self.scale = float(alpha) / self.rank
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.lora_a = nn.Linear(in_features, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int, alpha: float) -> "LoRALinear":
        m = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            rank,
            alpha,
        )
        m.linear.weight.data.copy_(linear.weight.data)
        if linear.bias is not None and m.linear.bias is not None:
            m.linear.bias.data.copy_(linear.bias.data)
        for p in m.linear.parameters():
            p.requires_grad = False
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.scale * self.lora_b(self.lora_a(x))


class DoRALinear(nn.Module):
    """Frozen Linear + LoRA branch scaled by learnable per-output magnitude (DoRA-style)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = max(1, int(rank))
        self.scale = float(alpha) / self.rank
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.lora_a = nn.Linear(in_features, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, out_features, bias=False)
        self.magnitude = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int, alpha: float) -> "DoRALinear":
        m = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            rank,
            alpha,
        )
        m.linear.weight.data.copy_(linear.weight.data)
        if linear.bias is not None and m.linear.bias is not None:
            m.linear.bias.data.copy_(linear.bias.data)
        for p in m.linear.parameters():
            p.requires_grad = False
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        delta = self.scale * self.lora_b(self.lora_a(x))
        # small bounded modulation of the low-rank path (magnitude init 0 → start = base)
        gate = torch.tanh(self.magnitude).view(1, -1)
        return base + gate * delta


def _normalize_targets(targets: Sequence[str]) -> Set[str]:
    out: Set[str] = set()
    for t in targets:
        out.add(t.strip().lower().replace("-", "_"))
    return out


def inject_linear_personalization(
    model: nn.Module,
    *,
    method: PersonalizationMethod,
    rank: int,
    alpha: float,
    targets: Sequence[str],
) -> List[str]:
    """
    Replace selected nn.Linear modules on ``xLSTMExerciseClassifier`` with LoRA / DoRA wrappers.

    Recognized target names:
      - ``input_proj`` — first map from kinematics to hidden dim
      - ``class_in`` — first Linear inside ``class_head`` (before GELU)

    Returns list of module names that were replaced (for logging / checkpoint keys).
    """
    from fitness_coach.models.xlstm_model import xLSTMExerciseClassifier

    if not isinstance(model, xLSTMExerciseClassifier):
        raise TypeError("inject_linear_personalization expects xLSTMExerciseClassifier")

    tset = _normalize_targets(targets)
    replaced: List[str] = []

    if "input_proj" in tset:
        old = model.xlstm.input_proj
        if isinstance(old, (LoRALinear, DoRALinear)):
            raise RuntimeError("input_proj already personalized")
        if method == "lora":
            model.xlstm.input_proj = LoRALinear.from_linear(old, rank, alpha)
        elif method == "dora":
            model.xlstm.input_proj = DoRALinear.from_linear(old, rank, alpha)
        else:
            raise ValueError("bottleneck does not use inject_linear_personalization")
        replaced.append("xlstm.input_proj")

    if "class_in" in tset:
        layer0 = model.class_head[0]
        if not isinstance(layer0, nn.Linear):
            raise TypeError("class_head[0] must be nn.Linear")
        if isinstance(layer0, (LoRALinear, DoRALinear)):
            raise RuntimeError("class_head[0] already personalized")
        if method == "lora":
            model.class_head[0] = LoRALinear.from_linear(layer0, rank, alpha)
        elif method == "dora":
            model.class_head[0] = DoRALinear.from_linear(layer0, rank, alpha)
        else:
            raise ValueError("bottleneck does not use inject_linear_personalization")
        replaced.append("class_head.0")

    if not replaced:
        raise ValueError(f"No layers matched targets={list(targets)}; use input_proj and/or class_in")
    return replaced


def trainable_personalization_params(model: nn.Module) -> List[nn.Parameter]:
    """All parameters marked ``requires_grad`` after ``freeze_base_unfreeze_personalization``."""
    return [p for p in model.parameters() if p.requires_grad]


def state_dict_personalization_only(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Subset of state_dict for LoRA/DoRA/BottleneckAdapter tensors (not frozen base linear weights)."""
    from fitness_coach.models.xlstm_model import xLSTMExerciseClassifier

    if not isinstance(model, xLSTMExerciseClassifier):
        raise TypeError("expected xLSTMExerciseClassifier")

    sd = model.state_dict()
    keep: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if "lora_a" in k or "lora_b" in k or ".magnitude" in k or k.startswith("adapter."):
            keep[k] = v
    return keep


def load_personalization_state_dict(model: nn.Module, payload: Dict[str, object], strict: bool = False) -> None:
    """
    Load personalization weights after base checkpoint is loaded and the same inject/config
    was applied as when the file was saved.
    """
    if "personalization_state_dict" in payload:
        model.load_state_dict(payload["personalization_state_dict"], strict=strict)
        return
    # Legacy bottleneck-only file
    if "adapter" in payload and isinstance(payload["adapter"], dict):
        from fitness_coach.models.xlstm_model import xLSTMExerciseClassifier

        if isinstance(model, xLSTMExerciseClassifier) and model.adapter is not None:
            model.adapter.load_state_dict(payload["adapter"], strict=strict)
            return
    raise KeyError("checkpoint must contain 'personalization_state_dict' or legacy 'adapter'")


def apply_personalization_from_file(model: nn.Module, path: Union[str, Path], *, map_location: str = "cpu") -> None:
    """
    Load a personalization checkpoint onto a base-loaded ``xLSTMExerciseClassifier``.
    Recreates LoRA/DoRA/Bottleneck structure from metadata then loads saved tensors.
    """
    p = Path(path)
    payload = torch.load(p, map_location=map_location, weights_only=False)
    method = str(payload.get("personalization_method", payload.get("method", "bottleneck"))).lower()
    if method in ("lora", "dora"):
        prepare_model_for_personalization(
            model,
            method=method,
            lora_rank=int(payload.get("lora_rank", 8)),
            lora_alpha=float(payload.get("lora_alpha", 16.0)),
            lora_targets=tuple(payload.get("lora_targets", ["input_proj", "class_in"])),
            adapter_dim=0,
        )
    elif method == "bottleneck":
        from fitness_coach.models.xlstm_model import BottleneckAdapter

        adim = int(payload.get("adapter_dim", 32))
        dr = float(payload.get("dropout", getattr(model, "dropout_rate", 0.1)))
        model.attach_adapter(
            BottleneckAdapter(model.hidden_size, bottleneck_dim=adim, dropout=dr)
        )
        freeze_base_unfreeze_personalization(model)
    else:
        raise ValueError(f"Unknown personalization_method={method!r}")
    load_personalization_state_dict(model, payload, strict=False)


def build_personalization_checkpoint(
    *,
    user_id: str,
    base_checkpoint: str,
    method: str,
    model: nn.Module,
    lora_rank: Optional[int] = None,
    lora_alpha: Optional[float] = None,
    targets: Optional[List[str]] = None,
    extra_meta: Optional[dict] = None,
) -> Dict[str, object]:
    meta = {
        "user_id": user_id,
        "base_checkpoint": base_checkpoint,
        "personalization_method": method,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_targets": targets,
    }
    if extra_meta:
        meta.update(extra_meta)
    meta["personalization_state_dict"] = state_dict_personalization_only(model)
    return meta


def freeze_base_unfreeze_personalization(model: nn.Module) -> None:
    """Freeze all params, then enable grads only on LoRA/DoRA/BottleneckAdapter."""
    from fitness_coach.models.xlstm_model import BottleneckAdapter, xLSTMExerciseClassifier

    for p in model.parameters():
        p.requires_grad = False
    if not isinstance(model, xLSTMExerciseClassifier):
        for p in model.parameters():
            p.requires_grad = True
        return
    for mod in model.modules():
        if isinstance(mod, (LoRALinear, DoRALinear)):
            for p in mod.lora_a.parameters():
                p.requires_grad = True
            for p in mod.lora_b.parameters():
                p.requires_grad = True
            if isinstance(mod, DoRALinear):
                mod.magnitude.requires_grad = True
        if isinstance(mod, BottleneckAdapter):
            for p in mod.parameters():
                p.requires_grad = True


def prepare_model_for_personalization(
    model: nn.Module,
    *,
    method: PersonalizationMethod,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_targets: Sequence[str] = ("input_proj", "class_in"),
    adapter_dim: int = 0,
) -> List[str]:
    """
    Attach trainable personalization modules. For ``bottleneck``, uses existing BottleneckAdapter
    API on the classifier (adapter_dim > 0 required).
    """
    from fitness_coach.models.xlstm_model import BottleneckAdapter, xLSTMExerciseClassifier

    if not isinstance(model, xLSTMExerciseClassifier):
        raise TypeError("expected xLSTMExerciseClassifier")

    replaced: List[str] = []
    if method == "bottleneck":
        if adapter_dim <= 0:
            raise ValueError("bottleneck personalization requires adapter_dim > 0")
        dim = model.hidden_size
        model.attach_adapter(
            BottleneckAdapter(dim, bottleneck_dim=adapter_dim, dropout=model.dropout_rate)
        )
        replaced.append("adapter")
        freeze_base_unfreeze_personalization(model)
        return replaced

    if method in ("lora", "dora"):
        replaced = inject_linear_personalization(
            model,
            method=method,
            rank=lora_rank,
            alpha=lora_alpha,
            targets=lora_targets,
        )
        freeze_base_unfreeze_personalization(model)
        return replaced

    raise ValueError(f"unknown method={method!r}")
