from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from concept_attention.config import load_config, save_resolved_config
from concept_attention.concepts import concepts_to_texts_and_groups, load_concepts_from_csv
from concept_attention.data import load_samples_from_csv, make_dataloader
from concept_attention.interpretation import global_concept_ranking
from concept_attention.losses import GroupLassoConfig, MentionRegConfig, prepare_group_index_tensors
from concept_attention.models import BlackBoxLM, MentionAlignedAVOHead, MentionAlignedAVOModel, build_concept_embeddings
from concept_attention.training import infer_task_and_outputs, train_blackbox, train_concept_model
from concept_attention.utils import ensure_dir, get_device, save_json, set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the concept-attention clinical text experiment.")
    parser.add_argument("--config", type=str, default="configs/mimic_sapbert.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = ensure_dir(cfg["paths"]["output_dir"])
    save_resolved_config(cfg, out_dir / "resolved_config.yaml")

    set_global_seed(cfg["training"]["seed"], cfg["training"]["deterministic"])
    device = get_device()
    print(f"[info] device={device}")

    train_samples = load_samples_from_csv(
        cfg["paths"]["train_csv"],
        cfg["data"]["text_col"],
        cfg["data"]["label_col"],
    )
    eval_samples = load_samples_from_csv(
        cfg["paths"]["eval_csv"],
        cfg["data"]["text_col"],
        cfg["data"]["label_col"],
    )
    print(f"[info] train samples={len(train_samples)}, eval samples={len(eval_samples)}")

    concepts = load_concepts_from_csv(cfg["paths"]["concept_csv"], **cfg["concepts"])
    concept_texts, groups, group_names, concept_ids = concepts_to_texts_and_groups(concepts)
    print(f"[info] real concepts={len(concept_texts)}, groups={len(groups)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["model_name"])
    encoder = AutoModel.from_pretrained(cfg["model"]["model_name"]).to(device)

    train_loader = make_dataloader(
        train_samples,
        tokenizer,
        batch_size=cfg["model"]["batch_size"],
        shuffle=True,
        max_length=cfg["model"]["max_length"],
        num_workers=cfg["model"]["num_workers"],
    )
    eval_loader = make_dataloader(
        eval_samples,
        tokenizer,
        batch_size=cfg["model"]["batch_size"],
        shuffle=False,
        max_length=cfg["model"]["max_length"],
        num_workers=cfg["model"]["num_workers"],
    )

    task, num_outputs = infer_task_and_outputs(train_samples)
    print(f"[info] task={task}, num_outputs={num_outputs}")

    blackbox = BlackBoxLM(encoder, num_outputs=num_outputs).to(device)
    warmup_history = train_blackbox(
        blackbox,
        train_loader,
        eval_loader,
        task=task,
        device=device,
        lr=cfg["training"]["blackbox_lr"],
        epochs=cfg["training"]["warmup_epochs"],
    )

    concept_emb = build_concept_embeddings(
        concept_texts=concept_texts,
        tokenizer=tokenizer,
        text_encoder=blackbox.encoder,
        device=device,
        batch_size=cfg["model"]["concept_batch_size"],
        max_length=cfg["model"]["concept_max_length"],
        pooling=cfg["model"]["concept_pooling"],
    ).to(device)

    head = MentionAlignedAVOHead(
        concept_emb=concept_emb,
        dv=cfg["model"]["dv"],
        num_outputs=num_outputs,
        temperature=cfg["model"]["temperature"],
        gate_margin=cfg["model"]["gate_margin"],
        gate_tau=cfg["model"]["gate_tau"],
        null_bias_init=cfg["model"]["null_bias_init"],
        top_k=min(int(cfg["model"]["top_k"]), len(concept_texts)),
        attn_activation=cfg["model"]["attention_activation"],
        freeze_concepts=cfg["model"]["freeze_concepts"],
        use_bias=True,
    ).to(device)

    concept_model = MentionAlignedAVOModel(
        text_encoder=blackbox.encoder,
        head=head,
        special_token_ids=getattr(tokenizer, "all_special_ids", []),
        freeze_text_encoder=cfg["model"]["freeze_text_encoder"],
    ).to(device)

    mention_cfg = MentionRegConfig(
        lambda_null_target=cfg["training"]["lambda_null_target"],
        null_target=cfg["training"]["null_target"],
        lambda_entropy=cfg["training"]["lambda_entropy"],
    )
    group_lasso_cfg = GroupLassoConfig(lambda_group_lasso=cfg["training"]["lambda_group_lasso"])
    group_row_indices = prepare_group_index_tensors(groups, device=device)

    concept_history = train_concept_model(
        concept_model,
        train_loader,
        eval_loader,
        task=task,
        device=device,
        group_row_indices=group_row_indices,
        mention_reg_cfg=mention_cfg,
        group_lasso_cfg=group_lasso_cfg,
        lr=cfg["training"]["concept_lr"],
        epochs=cfg["training"]["concept_epochs"],
    )

    metrics = {
        "task": task,
        "num_outputs": num_outputs,
        "warmup_history": warmup_history,
        "concept_history": concept_history,
    }
    save_json(metrics, out_dir / "metrics.json")

    torch.save(
        {
            "model_state_dict": blackbox.state_dict(),
            "model_name": cfg["model"]["model_name"],
            "num_outputs": num_outputs,
            "task": task,
        },
        out_dir / "blackbox.pt",
    )
    torch.save(
        {
            "model_state_dict": concept_model.state_dict(),
            "model_name": cfg["model"]["model_name"],
            "num_outputs": num_outputs,
            "task": task,
            "concept_texts": concept_texts,
            "concept_ids": concept_ids,
            "groups": groups,
            "group_names": group_names,
            "head_config": {
                "dv": cfg["model"]["dv"],
                "temperature": cfg["model"]["temperature"],
                "gate_margin": cfg["model"]["gate_margin"],
                "gate_tau": cfg["model"]["gate_tau"],
                "top_k": min(int(cfg["model"]["top_k"]), len(concept_texts)),
                "attn_activation": cfg["model"]["attention_activation"],
                "freeze_concepts": cfg["model"]["freeze_concepts"],
                "null_bias_init": cfg["model"]["null_bias_init"],
            },
        },
        out_dir / "concept_model.pt",
    )

    global_df = global_concept_ranking(concept_model, concept_ids, concept_texts)
    global_df.to_csv(out_dir / "global_concepts.csv", index=False)
    print(f"[done] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
