from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .losses import (
    GroupLassoConfig,
    MentionRegConfig,
    group_lasso_penalty_on_beta,
    mention_regularizers_avo,
)
from .metrics import compute_auc_aupr
from .utils import move_tensor_batch_to_device


def infer_task_and_outputs(train_samples: list[dict[str, Any]]) -> tuple[str, int]:
    """Infer multiclass, multilabel, or regression from the label format."""
    y0 = train_samples[0]["label"]

    if isinstance(y0, (list, tuple, np.ndarray, torch.Tensor)) and not (
        isinstance(y0, torch.Tensor) and y0.ndim == 0
    ):
        if isinstance(y0, torch.Tensor):
            return "multilabel", int(y0.numel())
        if isinstance(y0, np.ndarray):
            return "multilabel", int(y0.size)
        return "multilabel", len(y0)

    if isinstance(y0, (float, np.floating)) and not float(y0).is_integer():
        return "regression", 1

    labels = [int(s["label"]) for s in train_samples]
    return "multiclass", int(max(labels)) + 1


def get_loss_fn(task: str) -> nn.Module:
    if task == "multiclass":
        return nn.CrossEntropyLoss()
    if task == "multilabel":
        return nn.BCEWithLogitsLoss()
    if task == "regression":
        return nn.MSELoss()
    raise ValueError(f"Unknown task: {task}")


@torch.no_grad()
def evaluate_concept_model(model: nn.Module, val_loader: Any, task: str, device: torch.device) -> dict[str, float]:
    model.eval()
    all_probs, all_true = [], []

    for batch in val_loader:
        batch = move_tensor_batch_to_device(batch, device)
        out, _ = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids", None),
        )
        y = batch["labels"]

        if task == "multiclass":
            all_probs.append(torch.softmax(out.logits, dim=-1).detach().cpu().numpy())
            all_true.append(y.detach().cpu().numpy().astype(np.int64))
        elif task == "multilabel":
            all_probs.append(torch.sigmoid(out.logits).detach().cpu().numpy())
            all_true.append(y.detach().cpu().numpy())

    return _metrics_from_arrays(all_true, all_probs, task)


@torch.no_grad()
def evaluate_blackbox(model: nn.Module, val_loader: Any, task: str, device: torch.device) -> dict[str, float]:
    model.eval()
    all_probs, all_true = [], []

    for batch in val_loader:
        batch = move_tensor_batch_to_device(batch, device)
        logits, _ = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids", None),
        )
        y = batch["labels"]

        if task == "multiclass":
            all_probs.append(torch.softmax(logits, dim=-1).detach().cpu().numpy())
            all_true.append(y.detach().cpu().numpy().astype(np.int64))
        elif task == "multilabel":
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_true.append(y.detach().cpu().numpy())

    return _metrics_from_arrays(all_true, all_probs, task)


def _metrics_from_arrays(all_true: list[np.ndarray], all_probs: list[np.ndarray], task: str) -> dict[str, float]:
    if task == "regression":
        return {"note": "Regression task: AUROC/AUPR not applicable."}

    y_true = np.concatenate(all_true, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    metrics: dict[str, float] = {}

    if task == "multiclass":
        if y_prob.shape[1] == 2:
            metrics.update(compute_auc_aupr("binary1", y_true, y_prob[:, 1]))
        metrics.update(compute_auc_aupr("multiclass", y_true, y_prob))
    elif task == "multilabel":
        metrics.update(compute_auc_aupr("multilabel", y_true, y_prob))
    return metrics


def train_blackbox(
    model: nn.Module,
    train_loader: Any,
    val_loader: Any,
    *,
    task: str,
    device: torch.device,
    lr: float = 1e-6,
    epochs: int = 9,
) -> list[dict[str, Any]]:
    """Task-adaptive warmup: train the encoder with a CLS linear head."""
    loss_fn = get_loss_fn(task)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"warmup epoch {epoch}"):
            batch = move_tensor_batch_to_device(batch, device)
            logits, _ = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch.get("token_type_ids", None),
            )
            labels = batch["labels"]

            if task == "multiclass":
                loss = loss_fn(logits, labels.long())
            elif task == "multilabel":
                loss = loss_fn(logits, labels.float())
            else:
                loss = loss_fn(logits.squeeze(-1), labels.float())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            n_batches += 1

        metrics = evaluate_blackbox(model, val_loader, task, device)
        record = {"epoch": epoch, "train_loss": total_loss / max(n_batches, 1), **metrics}
        history.append(record)
        print(f"[warmup] epoch={epoch} " + " ".join(f"{k}={v}" for k, v in record.items() if k != "epoch"))

    return history


def train_concept_model(
    model: nn.Module,
    train_loader: Any,
    val_loader: Any,
    *,
    task: str,
    device: torch.device,
    group_row_indices: list[torch.Tensor],
    mention_reg_cfg: MentionRegConfig | None = None,
    group_lasso_cfg: GroupLassoConfig | None = None,
    lr: float = 5e-7,
    epochs: int = 1,
) -> list[dict[str, Any]]:
    loss_fn = get_loss_fn(task)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"concept epoch {epoch}"):
            batch = move_tensor_batch_to_device(batch, device)
            out, token_mask = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids", None),
            )
            labels = batch["labels"]

            if task == "multiclass":
                pred_loss = loss_fn(out.logits, labels.long())
            elif task == "multilabel":
                pred_loss = loss_fn(out.logits, labels.float())
            else:
                pred_loss = loss_fn(out.logits.squeeze(-1), labels.float())

            reg_loss = out.logits.new_zeros(())
            if mention_reg_cfg is not None:
                reg_loss = mention_regularizers_avo(out, token_mask, mention_reg_cfg)["loss_reg_total"]

            gl_loss = out.logits.new_zeros(())
            if group_lasso_cfg is not None and group_lasso_cfg.lambda_group_lasso > 0.0:
                beta = out.V @ out.O
                gl_loss = group_lasso_penalty_on_beta(beta, group_row_indices, group_lasso_cfg)

            loss = pred_loss + reg_loss + gl_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            n_batches += 1

        metrics = evaluate_concept_model(model, val_loader, task, device)
        record = {"epoch": epoch, "train_loss": total_loss / max(n_batches, 1), **metrics}
        history.append(record)
        print(f"[concept] epoch={epoch} " + " ".join(f"{k}={v}" for k, v in record.items() if k != "epoch"))

    return history
