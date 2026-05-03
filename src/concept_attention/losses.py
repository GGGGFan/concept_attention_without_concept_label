from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .models import AVOOutput


@dataclass
class MentionRegConfig:
    lambda_null_target: float = 0.02
    null_target: float = 0.95
    lambda_entropy: float = 0.02
    eps: float = 1e-8


@dataclass
class GroupLassoConfig:
    lambda_group_lasso: float = 1e-3
    use_sqrt_group_size_weight: bool = True
    eps: float = 1e-8


def mention_regularizers_avo(
    out: AVOOutput,
    token_mask: torch.Tensor | None,
    cfg: MentionRegConfig,
) -> dict[str, torch.Tensor]:
    """Notebook regularizers that encourage most tokens to align with NULL and low real-concept entropy."""
    A = out.A
    A_null = A[..., 0]
    A_real = A[..., 1:]

    mean_null = A_null[token_mask].mean() if token_mask is not None else A_null.mean()
    loss_null = cfg.lambda_null_target * (mean_null - cfg.null_target) ** 2

    denom = (1.0 - A_null).clamp(min=cfg.eps).unsqueeze(-1)
    p_real = (A_real / denom).clamp(min=cfg.eps)
    entropy = -(p_real * p_real.log()).sum(dim=-1)

    w = (1.0 - A_null).detach()
    if token_mask is not None:
        mask = token_mask.float()
        denom_w = (mask * w).sum().clamp(min=1.0)
        ent = (entropy * mask * w).sum() / denom_w
    else:
        ent = (entropy * w).sum() / w.sum().clamp(min=1.0)

    loss_entropy = cfg.lambda_entropy * ent
    total = loss_null + loss_entropy
    return {
        "loss_null_target": loss_null,
        "loss_entropy": loss_entropy,
        "loss_reg_total": total,
    }


def prepare_group_index_tensors(groups: list[list[int]], device: torch.device) -> list[torch.Tensor]:
    """Shift concept indices by +1 because beta row 0 is the internally added NULL concept."""
    tensors: list[torch.Tensor] = []
    for group in groups:
        if group:
            tensors.append(torch.tensor(group, dtype=torch.long, device=device) + 1)
    return tensors


def group_lasso_penalty_on_beta(
    beta_with_null: torch.Tensor,
    group_row_indices: list[torch.Tensor],
    cfg: GroupLassoConfig,
) -> torch.Tensor:
    """Plain group lasso on beta = V @ O, excluding the NULL row."""
    if cfg.lambda_group_lasso <= 0.0:
        return beta_with_null.new_zeros(())

    penalty = beta_with_null.new_zeros(())
    for idx in group_row_indices:
        beta_group = beta_with_null.index_select(0, idx)
        frob = torch.sqrt((beta_group * beta_group).sum() + cfg.eps)
        weight = math.sqrt(float(idx.numel())) if cfg.use_sqrt_group_size_weight else 1.0
        penalty = penalty + weight * frob

    return cfg.lambda_group_lasso * penalty
