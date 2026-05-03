from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor | None, dim: int) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=dim)
    mask = mask.bool()
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    mask_f = mask.to(x.dtype)
    denom = mask_f.sum(dim=dim).clamp(min=1.0)
    return (x * mask_f).sum(dim=dim) / denom


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax activation used in the notebook to obtain sparse token-concept attention."""
    if dim < 0:
        dim = logits.dim() + dim

    z = logits.transpose(dim, -1)
    orig_shape = z.shape
    z = z.reshape(-1, orig_shape[-1])

    z_sorted, _ = torch.sort(z, descending=True, dim=-1)
    z_cumsum = z_sorted.cumsum(dim=-1)

    k = torch.arange(1, z.shape[-1] + 1, device=z.device, dtype=z.dtype).view(1, -1)
    cond = 1 + k * z_sorted > z_cumsum
    k_z = cond.sum(dim=-1, keepdim=True).clamp(min=1)

    idx = (k_z - 1).long()
    tau = (z_cumsum.gather(dim=-1, index=idx) - 1) / k_z.to(z.dtype)

    p = torch.clamp(z - tau, min=0.0)
    return p.reshape(orig_shape).transpose(dim, -1)


@torch.no_grad()
def build_concept_embeddings(
    concept_texts: list[str],
    tokenizer: Any,
    text_encoder: nn.Module,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 32,
    pooling: Literal["cls", "mean"] = "cls",
) -> torch.Tensor:
    """Encode concept descriptions using the same warmed-up encoder as the notes."""
    text_encoder.eval()
    embs: list[torch.Tensor] = []

    for start in range(0, len(concept_texts), batch_size):
        batch = concept_texts[start : start + batch_size]
        tok = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        out = text_encoder(**tok)
        h = out.last_hidden_state

        if pooling == "cls":
            emb = h[:, 0, :]
        elif pooling == "mean":
            mask = tok["attention_mask"].unsqueeze(-1)
            emb = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            raise ValueError("pooling must be 'cls' or 'mean'.")

        embs.append(emb.detach().cpu())

    return torch.cat(embs, dim=0)


@dataclass
class AVOOutput:
    logits: torch.Tensor
    token_logits: torch.Tensor
    A: torch.Tensor
    V: torch.Tensor
    O: torch.Tensor
    sim: torch.Tensor
    A_pool: torch.Tensor | None = None
    AV_pool: torch.Tensor | None = None


class BlackBoxLM(nn.Module):
    """Warmup/blackbox model from the notebook: encoder CLS embedding + linear head."""

    def __init__(self, encoder: nn.Module, num_outputs: int):
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size
        self.head = nn.Linear(hidden_size, num_outputs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        cls = out.last_hidden_state[:, 0, :]
        return self.head(cls), cls


class MentionAlignedAVOHead(nn.Module):
    """
    Mention-aligned AVO head.

    A has shape (B, L, C + 1); A[..., 0] is the NULL concept.
    V has shape (C + 1, dv); V[0] is zero, so the NULL concept contributes no value.
    Final logits are computed as A_pool @ V @ O (+ bias).
    """

    def __init__(
        self,
        concept_emb: torch.Tensor,
        dv: int,
        num_outputs: int,
        temperature: float = 0.07,
        gate_margin: float = 0.85,
        gate_tau: float = 0.05,
        top_k: int | None = 8,
        attn_activation: Literal["softmax", "sparsemax"] = "softmax",
        freeze_concepts: bool = True,
        use_bias: bool = True,
        null_bias_init: float = 0.0,
    ):
        super().__init__()
        if concept_emb.ndim != 2:
            raise ValueError("concept_emb must have shape (C, H).")

        n_concepts, hidden_size = concept_emb.shape
        self.C = n_concepts
        self.H = hidden_size
        self.dv = dv
        self.num_outputs = num_outputs
        self.temperature = float(temperature)
        self.gate_margin = float(gate_margin)
        self.gate_tau = float(gate_tau)
        self.top_k = top_k
        self.attn_activation = attn_activation

        if freeze_concepts:
            self.register_buffer("concept_emb", concept_emb.detach().clone())
        else:
            self.concept_emb = nn.Parameter(concept_emb.detach().clone())

        self.Wv = nn.Linear(hidden_size, dv, bias=False)
        self.O = nn.Parameter(torch.randn(dv, num_outputs) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_outputs)) if use_bias else None
        self.null_bias = nn.Parameter(torch.tensor(float(null_bias_init)))

    def _attn(self, logits_full: torch.Tensor) -> torch.Tensor:
        if self.attn_activation == "softmax":
            return torch.softmax(logits_full, dim=-1)
        if self.attn_activation == "sparsemax":
            return sparsemax(logits_full, dim=-1)
        raise ValueError("attn_activation must be 'softmax' or 'sparsemax'.")

    def concept_value_matrix(self) -> torch.Tensor:
        v_real = self.Wv(self.concept_emb)
        v_null = v_real.new_zeros(1, self.dv)
        return torch.cat([v_null, v_real], dim=0)

    def beta(self) -> torch.Tensor:
        return self.concept_value_matrix() @ self.O

    def forward(self, token_embs: torch.Tensor, token_mask: torch.Tensor | None = None) -> AVOOutput:
        batch_size, seq_len, hidden_size = token_embs.shape
        if hidden_size != self.H:
            raise ValueError(f"token_embs dim {hidden_size} must match concept_emb dim {self.H}.")

        q = F.normalize(token_embs, p=2, dim=-1)
        k = F.normalize(self.concept_emb, p=2, dim=-1)
        sim = torch.einsum("blh,ch->blc", q, k)

        logits_real = sim / max(self.temperature, 1e-6)

        if self.top_k is not None and self.top_k < self.C:
            _, idx = torch.topk(logits_real, k=self.top_k, dim=-1)
            keep = torch.zeros_like(logits_real, dtype=torch.bool)
            keep.scatter_(-1, idx, True)
            logits_real = logits_real.masked_fill(~keep, -1e9)

        s_max = sim.max(dim=-1, keepdim=True).values
        null_logit = (self.gate_margin - s_max) / max(self.gate_tau, 1e-6)
        null_logit = null_logit + self.null_bias
        logits_full = torch.cat([null_logit, logits_real], dim=-1)

        if token_mask is not None:
            token_mask = token_mask.bool()
            masked = ~token_mask
            logits_full = logits_full.masked_fill(masked.unsqueeze(-1), -1e9)
            logits_full[..., 0] = logits_full[..., 0].masked_fill(masked, 0.0)

        A = self._attn(logits_full)
        V = self.concept_value_matrix()

        # The notebook used max pooling over token-level attention scores.
        if token_mask is not None:
            A_masked = A.masked_fill(~token_mask.bool().unsqueeze(-1), 0.0)
            A_pool = A_masked.max(dim=1).values
        else:
            A_pool = A.max(dim=1).values

        AV_pool = torch.einsum("bc,cd->bd", A_pool, V)
        logits = torch.einsum("bd,do->bo", AV_pool, self.O)
        if self.bias is not None:
            logits = logits + self.bias

        return AVOOutput(
            logits=logits,
            token_logits=logits,
            A=A,
            V=V,
            O=self.O,
            sim=sim,
            A_pool=A_pool,
            AV_pool=AV_pool,
        )


class MentionAlignedAVOModel(nn.Module):
    """HuggingFace encoder wrapper plus the concept-attention head."""

    def __init__(
        self,
        text_encoder: nn.Module,
        head: MentionAlignedAVOHead,
        special_token_ids: list[int] | None = None,
        freeze_text_encoder: bool = True,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.head = head
        self.special_token_ids = special_token_ids or []

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        # The notebook froze token/position/type embeddings even when the encoder remained trainable.
        embeddings = getattr(self.text_encoder, "embeddings", None)
        if embeddings is not None:
            for name in ["word_embeddings", "position_embeddings", "token_type_embeddings", "LayerNorm"]:
                module = getattr(embeddings, name, None)
                if module is not None:
                    for p in module.parameters():
                        p.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> tuple[AVOOutput, torch.Tensor]:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        enc_out = self.text_encoder(**kwargs)
        token_embs = enc_out.last_hidden_state

        token_mask = attention_mask.bool()
        for tid in self.special_token_ids:
            token_mask = token_mask & (input_ids != tid)

        return self.head(token_embs=token_embs, token_mask=token_mask), token_mask
