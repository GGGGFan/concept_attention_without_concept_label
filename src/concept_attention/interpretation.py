from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .models import MentionAlignedAVOModel


@torch.no_grad()
def beta_matrix(model: MentionAlignedAVOModel) -> torch.Tensor:
    """Return beta = V @ O with row 0 corresponding to the NULL concept."""
    return model.head.beta().detach()


@torch.no_grad()
def global_concept_ranking(
    model: MentionAlignedAVOModel,
    concept_ids: list[str],
    concept_texts: list[str],
    *,
    class_index: int = 1,
    contrast_negative_class: bool = True,
) -> pd.DataFrame:
    """
    Rank concepts by beta.

    For binary classification, the notebook inspected beta[:, 1] - beta[:, 0].
    """
    beta = beta_matrix(model).cpu().numpy()

    if beta.shape[1] == 2 and contrast_negative_class and class_index == 1:
        score = beta[:, 1] - beta[:, 0]
        score_name = "beta_class1_minus_class0"
    else:
        score = beta[:, class_index]
        score_name = f"beta_class{class_index}"

    rows = []
    for j, (cid, ctext) in enumerate(zip(concept_ids, concept_texts), start=1):
        rows.append(
            {
                "concept_row": j,
                "concept_id": cid,
                "concept_text": ctext,
                score_name: float(score[j]),
            }
        )
    return pd.DataFrame(rows).sort_values(score_name, ascending=False).reset_index(drop=True)


@torch.no_grad()
def local_concept_contributions(
    out,
    *,
    class_index: int = 1,
    contrast_negative_class: bool = False,
) -> torch.Tensor:
    """
    Return per-sample concept contributions A_pool * beta for the selected class.

    Shape: (B, C + 1), including the NULL row.
    """
    beta = out.V @ out.O
    if beta.shape[1] == 2 and contrast_negative_class and class_index == 1:
        coef = beta[:, 1] - beta[:, 0]
    else:
        coef = beta[:, class_index]
    return out.A_pool * coef


def top_concepts_for_sample(
    contributions: torch.Tensor,
    concept_texts: list[str],
    sample_index: int = 0,
    top_n: int = 10,
) -> list[dict[str, float | int | str]]:
    """Return top positive concept contributions for one sample. Excludes NULL."""
    scores = contributions[sample_index].detach().cpu().numpy()
    order = np.argsort(scores)[::-1]
    rows = []
    for concept_row in order:
        if concept_row == 0:
            continue
        rows.append(
            {
                "concept_row": int(concept_row),
                "concept_text": concept_texts[int(concept_row) - 1],
                "contribution": float(scores[int(concept_row)]),
            }
        )
        if len(rows) >= top_n:
            break
    return rows
