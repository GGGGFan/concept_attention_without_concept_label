from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .metrics import binary_auc_roc, binary_aupr_ap
from .utils import move_tensor_batch_to_device


def pick_top_middle_bottom(ranked_list: list[int], k: int = 5) -> tuple[list[int], list[int], list[int]]:
    """Selection used in the notebook for top, neutral/middle, and bottom concepts."""
    n = len(ranked_list)
    top_k = ranked_list[:k]
    bottom_k = ranked_list[-k:]
    mid_start = max(0, (n - k) // 2)
    mid_k = ranked_list[mid_start : mid_start + k]
    return top_k, mid_k, bottom_k


@torch.no_grad()
def run_blackbox_concept_perturbation(
    concept_model: Any,
    blackbox_model: Any,
    val_loader: Any,
    *,
    device: torch.device,
    class_index: int = 1,
    k: int = 5,
) -> dict[str, dict[str, float]]:
    """
    Perturbation analysis from the notebook.

    For tokens linked to selected concepts, the notebook set token IDs to 0 and
    attention mask to 0 before passing the notes to the blackbox model.
    """
    concept_model.eval()
    blackbox_model.eval()

    y_true: list[int] = []
    pred_all: list[float] = []
    pred_top: list[float] = []
    pred_mid: list[float] = []
    pred_bottom: list[float] = []

    for batch in tqdm(val_loader, desc=f"perturbation top/mid/bottom k={k}"):
        batch = move_tensor_batch_to_device(batch, device)

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch.get("token_type_ids", None)
        labels = batch["labels"]

        out, _ = concept_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        beta = (out.V @ out.O)[:, class_index]
        contribution = out.A.max(1).values * beta

        input_ids_top = input_ids.clone()
        attention_mask_top = attention_mask.clone()
        input_ids_mid = input_ids.clone()
        attention_mask_mid = attention_mask.clone()
        input_ids_bottom = input_ids.clone()
        attention_mask_bottom = attention_mask.clone()

        for bi in range(input_ids.shape[0]):
            sorted_idx = torch.argsort(contribution[bi], descending=True).tolist()
            groups = pick_top_middle_bottom(sorted_idx, k=k)
            for group_id, concept_rows in enumerate(groups):
                for concept_row in concept_rows:
                    indices = [i for i, x in enumerate(out.A[bi, :, concept_row].tolist()) if x > 0]
                    if group_id == 0:
                        input_ids_top[bi][indices] = 0
                        attention_mask_top[bi][indices] = 0
                    elif group_id == 1:
                        input_ids_mid[bi][indices] = 0
                        attention_mask_mid[bi][indices] = 0
                    else:
                        input_ids_bottom[bi][indices] = 0
                        attention_mask_bottom[bi][indices] = 0

        logits, _ = blackbox_model(input_ids, attention_mask, token_type_ids)
        logits_top, _ = blackbox_model(input_ids_top, attention_mask_top, token_type_ids)
        logits_mid, _ = blackbox_model(input_ids_mid, attention_mask_mid, token_type_ids)
        logits_bottom, _ = blackbox_model(input_ids_bottom, attention_mask_bottom, token_type_ids)

        pred_all.extend(torch.softmax(logits, dim=-1)[:, class_index].detach().cpu().tolist())
        pred_top.extend(torch.softmax(logits_top, dim=-1)[:, class_index].detach().cpu().tolist())
        pred_mid.extend(torch.softmax(logits_mid, dim=-1)[:, class_index].detach().cpu().tolist())
        pred_bottom.extend(torch.softmax(logits_bottom, dim=-1)[:, class_index].detach().cpu().tolist())
        y_true.extend(labels.detach().cpu().tolist())

    y = np.asarray(y_true)
    results = {}
    for name, scores in {
        "blackbox": pred_all,
        "top_concepts_masked": pred_top,
        "neutral_concepts_masked": pred_mid,
        "bottom_concepts_masked": pred_bottom,
    }.items():
        s = np.asarray(scores)
        results[name] = {"AUROC": binary_auc_roc(y, s), "AUPR": binary_aupr_ap(y, s)}

    return results
