from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class TextLabelDataset(Dataset):
    """Dataset used by the original notebook: each sample has text (`txt`) and label."""

    def __init__(self, samples: list[dict[str, Any]]):
        self.samples = samples
        for i, sample in enumerate(self.samples[:5]):
            if "txt" not in sample or "label" not in sample:
                raise KeyError(f"Sample at index {i} is missing 'txt' or 'label'.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        return {"txt": sample["txt"], "label": sample["label"]}


def load_samples_from_csv(
    csv_path: str,
    text_col: str = "text",
    label_col: str = "hospital_expire_flag",
) -> list[dict[str, Any]]:
    """Load notes and labels from a CSV, matching the notebook's `train_samples` format."""
    df = pd.read_csv(csv_path)
    missing = [c for c in (text_col, label_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    samples: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        text = row[text_col]
        if isinstance(text, str):
            label = row[label_col]
            if isinstance(label, (np.integer, int, bool, np.bool_)):
                label = int(label)
            elif isinstance(label, (np.floating, float)) and float(label).is_integer():
                label = int(label)
            samples.append({"txt": text, "label": label})
    return samples


def _infer_label_tensor(labels: list[Any]) -> torch.Tensor:
    def to_py(x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist() if x.ndim > 0 else x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    labels_py = [to_py(x) for x in labels]
    first = labels_py[0]

    if isinstance(first, (list, tuple)):
        return torch.tensor(labels_py, dtype=torch.float)

    if isinstance(first, (bool, np.bool_, int, np.integer)):
        return torch.tensor(labels_py, dtype=torch.long)
    if isinstance(first, (float, np.floating)):
        return torch.tensor(labels_py, dtype=torch.float)

    tensor = torch.tensor(labels_py)
    if tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        return tensor.long()
    return tensor.float()


def make_collate_fn(tokenizer: Any, max_length: int = 256, return_text: bool = False) -> Callable:
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [b["txt"] for b in batch]
        labels = [b["label"] for b in batch]

        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        out = {**enc, "labels": _infer_label_tensor(labels)}
        if return_text:
            out["txt"] = texts
        return out

    return collate


def make_dataloader(
    samples: list[dict[str, Any]],
    tokenizer: Any,
    batch_size: int,
    shuffle: bool,
    max_length: int = 256,
    num_workers: int = 2,
    pin_memory: bool = True,
    return_text: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        TextLabelDataset(samples),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=make_collate_fn(tokenizer, max_length=max_length, return_text=return_text),
        drop_last=drop_last,
    )
