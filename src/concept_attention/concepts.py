from __future__ import annotations

from collections import OrderedDict
from typing import Any

import pandas as pd


def _load_optional_from_n3c() -> tuple[Any | None, Any | None]:
    try:
        from from_n3c import icd10_text, infer_chapter_from_code  # type: ignore

        return icd10_text, infer_chapter_from_code
    except Exception:
        return None, None


def infer_icd10_chapter(code: str) -> str:
    """Fallback ICD-10 chapter label when the original `from_n3c` helper is unavailable."""
    code = str(code).strip().upper()
    if not code:
        return "Unknown ICD-10 chapter"

    letter = code[0]
    try:
        number = int("".join(ch for ch in code[1:3] if ch.isdigit()) or 0)
    except ValueError:
        number = 0

    if letter in {"A", "B"}:
        return "Certain infectious and parasitic diseases"
    if letter == "C" or (letter == "D" and number <= 49):
        return "Neoplasms"
    if letter == "D":
        return "Diseases of the blood and immune mechanism"
    if letter == "E":
        return "Endocrine, nutritional and metabolic diseases"
    if letter == "F":
        return "Mental and behavioural disorders"
    if letter == "G":
        return "Diseases of the nervous system"
    if letter == "H" and number <= 59:
        return "Diseases of the eye and adnexa"
    if letter == "H":
        return "Diseases of the ear and mastoid process"
    if letter == "I":
        return "Diseases of the circulatory system"
    if letter == "J":
        return "Diseases of the respiratory system"
    if letter == "K":
        return "Diseases of the digestive system"
    if letter == "L":
        return "Diseases of the skin and subcutaneous tissue"
    if letter == "M":
        return "Diseases of the musculoskeletal system"
    if letter == "N":
        return "Diseases of the genitourinary system"
    if letter == "O":
        return "Pregnancy, childbirth and puerperium"
    if letter == "P":
        return "Certain conditions originating in the perinatal period"
    if letter == "Q":
        return "Congenital malformations"
    if letter == "R":
        return "Symptoms, signs and abnormal findings"
    if letter in {"S", "T"}:
        return "Injury, poisoning and certain other consequences of external causes"
    if letter in {"V", "W", "X", "Y"}:
        return "External causes of morbidity"
    if letter == "Z":
        return "Factors influencing health status"
    return "Unknown ICD-10 chapter"


def infer_ancestral_category(code: str) -> str:
    """Conservative fallback used only if the original helper/column is not available."""
    code = str(code).strip().upper()
    return code[:3] if len(code) >= 3 else code


def load_concepts_from_csv(
    csv_path: str,
    *,
    code_col: str = "code",
    name_col: str = "name",
    group_col: str = "idx_section",
    group_offset: int = 1,
    template: str = "{name} (Ancestral category: {ancestral_category}, {chapter})",
    use_from_n3c_if_available: bool = True,
) -> list[dict[str, str]]:
    """
    Load concept rows in the same structure used by the notebook.

    Expected default columns:
      - `code`
      - `name`
      - `idx_section`

    If the original `from_n3c` helpers are available, they are used to reproduce
    the concept text construction from the notebook.
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in (code_col, name_col, group_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required concept columns in {csv_path}: {missing}")

    icd10_text_fn, chapter_fn = (None, None)
    if use_from_n3c_if_available:
        icd10_text_fn, chapter_fn = _load_optional_from_n3c()

    concepts: list[dict[str, str]] = []
    for _, row in df.iterrows():
        code = str(row[code_col])
        name = str(row[name_col])

        ancestral_category = (
            str(icd10_text_fn(code))
            if icd10_text_fn is not None
            else infer_ancestral_category(code)
        )
        chapter = (
            str(chapter_fn(code))
            if chapter_fn is not None
            else infer_icd10_chapter(code)
        )

        group_value = row[group_col]
        if isinstance(group_value, (int, float)) and float(group_value).is_integer():
            group = str(int(group_value) + int(group_offset))
        else:
            group = str(group_value)

        text = template.format(
            code=code,
            name=name,
            ancestral_category=ancestral_category,
            chapter=chapter,
            group=group,
        )
        concepts.append({"id": code, "text": text, "group": group})

    return concepts


def concepts_to_texts_and_groups(
    concepts: list[dict[str, Any]],
    text_key: str = "text",
    group_key: str = "group",
) -> tuple[list[str], list[list[int]], list[str], list[str]]:
    """
    Returns concept texts, group index lists, group names, and concept IDs.

    Concept indices are for real concepts only. The model adds NULL internally.
    """
    concept_texts: list[str] = []
    concept_ids: list[str] = []
    group_to_indices: OrderedDict[str, list[int]] = OrderedDict()

    for i, concept in enumerate(concepts):
        if text_key not in concept:
            raise KeyError(f"Concept {i} is missing '{text_key}'.")
        concept_texts.append(str(concept[text_key]))
        concept_ids.append(str(concept.get("id", i)))

        if group_key not in concept:
            raise KeyError(f"Concept {i} is missing '{group_key}'.")
        group = str(concept[group_key])
        group_to_indices.setdefault(group, []).append(i)

    group_names = list(group_to_indices.keys())
    groups = [group_to_indices[g] for g in group_names]
    return concept_texts, groups, group_names, concept_ids
