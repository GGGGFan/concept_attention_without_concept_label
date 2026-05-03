from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from concept_attention.config import load_config
from concept_attention.data import load_samples_from_csv, make_dataloader
from concept_attention.losses import prepare_group_index_tensors
from concept_attention.models import BlackBoxLM, MentionAlignedAVOHead, MentionAlignedAVOModel, build_concept_embeddings
from concept_attention.perturbation import run_blackbox_concept_perturbation
from concept_attention.utils import get_device, save_json, set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run concept-based perturbation analysis.")
    parser.add_argument("--config", type=str, default="configs/mimic_sapbert.yaml")
    parser.add_argument("--k", type=int, default=5, help="Number of concepts to perturb: 5 or 10 in the manuscript.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["paths"]["output_dir"])

    set_global_seed(cfg["training"]["seed"], cfg["training"]["deterministic"])
    device = get_device()

    blackbox_ckpt = torch.load(out_dir / "blackbox.pt", map_location=device)
    concept_ckpt = torch.load(out_dir / "concept_model.pt", map_location=device)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["model_name"])
    encoder = AutoModel.from_pretrained(cfg["model"]["model_name"]).to(device)

    blackbox = BlackBoxLM(encoder, num_outputs=blackbox_ckpt["num_outputs"]).to(device)
    blackbox.load_state_dict(blackbox_ckpt["model_state_dict"])

    # Rebuild the concept model architecture with the same warmed encoder and concepts.
    concept_texts = concept_ckpt["concept_texts"]
    concept_emb = build_concept_embeddings(
        concept_texts=concept_texts,
        tokenizer=tokenizer,
        text_encoder=blackbox.encoder,
        device=device,
        batch_size=cfg["model"]["concept_batch_size"],
        max_length=cfg["model"]["concept_max_length"],
        pooling=cfg["model"]["concept_pooling"],
    ).to(device)

    head_cfg = concept_ckpt["head_config"]
    head = MentionAlignedAVOHead(
        concept_emb=concept_emb,
        dv=head_cfg["dv"],
        num_outputs=concept_ckpt["num_outputs"],
        temperature=head_cfg["temperature"],
        gate_margin=head_cfg["gate_margin"],
        gate_tau=head_cfg["gate_tau"],
        top_k=head_cfg["top_k"],
        attn_activation=head_cfg["attn_activation"],
        freeze_concepts=head_cfg["freeze_concepts"],
        null_bias_init=head_cfg["null_bias_init"],
        use_bias=True,
    ).to(device)
    concept_model = MentionAlignedAVOModel(
        text_encoder=blackbox.encoder,
        head=head,
        special_token_ids=getattr(tokenizer, "all_special_ids", []),
        freeze_text_encoder=cfg["model"]["freeze_text_encoder"],
    ).to(device)
    concept_model.load_state_dict(concept_ckpt["model_state_dict"])

    eval_samples = load_samples_from_csv(
        cfg["paths"]["eval_csv"],
        cfg["data"]["text_col"],
        cfg["data"]["label_col"],
    )
    eval_loader = make_dataloader(
        eval_samples,
        tokenizer,
        batch_size=cfg["model"]["batch_size"],
        shuffle=False,
        max_length=cfg["model"]["max_length"],
        num_workers=cfg["model"]["num_workers"],
    )

    results = run_blackbox_concept_perturbation(
        concept_model,
        blackbox,
        eval_loader,
        device=device,
        class_index=1,
        k=args.k,
    )
    save_json(results, out_dir / f"perturbation_top{args.k}.json")
    print(results)


if __name__ == "__main__":
    main()
