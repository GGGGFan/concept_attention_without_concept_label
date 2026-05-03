# Concept-Based Interpretable Clinical Text Prediction

This repository contains a cleaned and reproducible implementation of the notebook experiment for **concept-based interpretation of clinical text predictions via token-concept attention without concept supervision**.

The implementation follows the manuscript workflow: clinical notes and ICD-10 concept descriptions are embedded with the same pretrained encoder; token-concept attention estimates how strongly each concept is reflected in a note; pooled concept relevance scores are used in a linear concept-level prediction layer; global and local interpretation are derived from the learned concept coefficients and token-level attention scores.

## Repository layout

```text
.
├── configs/
│   └── mimic_sapbert.yaml
├── scripts/
│   ├── run_experiment.py
│   └── run_perturbation.py
├── src/
│   └── concept_attention/
│       ├── __init__.py
│       ├── config.py
│       ├── concepts.py
│       ├── data.py
│       ├── interpretation.py
│       ├── losses.py
│       ├── metrics.py
│       ├── models.py
│       ├── perturbation.py
│       ├── training.py
│       ├── utils.py
│       └── visualization.py
├── requirements.txt
├── .gitignore
└── README.md
```

## File descriptions

- `configs/mimic_sapbert.yaml`: paths, model settings, and hyperparameters taken from the notebook.
- `scripts/run_experiment.py`: main command-line entry point. It loads data, warms up the blackbox encoder, builds concept embeddings, trains the concept-attention model, evaluates it, and exports global concept rankings.
- `scripts/run_perturbation.py`: reproduces the notebook-style perturbation analysis by masking tokens linked to top, neutral, or bottom contributing concepts and evaluating the blackbox model.
- `src/concept_attention/data.py`: CSV loading, dataset, collate function, and dataloader utilities.
- `src/concept_attention/concepts.py`: ICD-10 concept loading and grouping. If `from_n3c` is available, it is used to reproduce the notebook concept strings.
- `src/concept_attention/models.py`: blackbox warmup model, Sparsemax, concept embedding extraction, and the mention-aligned AVO token-concept attention model.
- `src/concept_attention/losses.py`: NULL-attention regularization and group lasso on `beta = V @ O`.
- `src/concept_attention/training.py`: warmup training, concept-model training, and evaluation.
- `src/concept_attention/metrics.py`: AUROC and AUPR utilities.
- `src/concept_attention/interpretation.py`: global concept ranking and local contribution helpers.
- `src/concept_attention/perturbation.py`: perturbation evaluation logic.
- `src/concept_attention/visualization.py`: token-level attention rendering utilities.
- `requirements.txt`: minimal Python dependencies.
- `.gitignore`: excludes raw data, outputs, checkpoints, secrets, and local environment files.

## Expected input files

This repository does **not** include raw data.

The default config assumes three CSV-style inputs:

### Clinical note CSV

Used for both training and evaluation.

Required columns by default:

```text
text,hospital_expire_flag
```

You can rename these in `configs/mimic_sapbert.yaml`:

```yaml
data:
  text_col: text
  label_col: hospital_expire_flag
```

### ICD-10 concept CSV

Required columns by default:

```text
code,name,idx_section
```

The notebook created concept strings using:

```python
f"{name} (Ancestral category: {icd10_text(code)}, {infer_chapter_from_code(code)})"
```

This repository tries to import `from_n3c.icd10_text` and `from_n3c.infer_chapter_from_code` when available. If those helpers are not available, it falls back to conservative ICD-10 chapter text so that the code remains runnable. For exact reproduction, keep `from_n3c` available in your environment or provide equivalent concept-text columns and adjust `src/concept_attention/concepts.py`.

## Installation

```bash
git clone <your-repo-url>
cd <repo-name>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU use, install the PyTorch build appropriate for your CUDA version.

## Configure paths

Edit `configs/mimic_sapbert.yaml`:

```yaml
paths:
  train_csv: /path/to/mimic_processed_notes/MP_IN_train.csv
  eval_csv: /path/to/mimic_processed_notes/MP_IN_test.csv
  concept_csv: /path/to/df_icd10.csv
  output_dir: outputs/mimic_sapbert
```

The config preserves the main notebook hyperparameters:

```yaml
model:
  model_name: cambridgeltl/SapBERT-from-PubMedBERT-fulltext
  batch_size: 4
  max_length: 512
  dv: 256
  freeze_text_encoder: false
  freeze_concepts: true
  attention_activation: sparsemax
  top_k: 4
  temperature: 0.07
  gate_margin: 0.85
  gate_tau: 0.05
  null_bias_init: 2.0

training:
  seed: 42
  warmup_epochs: 9
  concept_epochs: 1
  blackbox_lr: 0.000001
  concept_lr: 0.0000005
  lambda_null_target: 0.02
  null_target: 0.95
  lambda_entropy: 0.02
  lambda_group_lasso: 0.001
```

## Run the main experiment

```bash
python scripts/run_experiment.py --config configs/mimic_sapbert.yaml
```

Outputs are written to `output_dir`:

```text
outputs/mimic_sapbert/
├── resolved_config.yaml
├── metrics.json
├── blackbox.pt
├── concept_model.pt
└── global_concepts.csv
```

`global_concepts.csv` ranks ICD-10 concepts using the binary-class contrast `beta[:, 1] - beta[:, 0]`, matching the notebook inspection code.

## Run perturbation analysis

After running the main experiment:

```bash
python scripts/run_perturbation.py --config configs/mimic_sapbert.yaml --k 5
python scripts/run_perturbation.py --config configs/mimic_sapbert.yaml --k 10
```

This writes:

```text
outputs/mimic_sapbert/perturbation_top5.json
outputs/mimic_sapbert/perturbation_top10.json
```

The perturbation script follows the notebook logic: concepts are ranked by `A.max(1) * beta`; tokens with nonzero attention to selected concepts are zeroed out with their attention mask set to 0 before the note is passed to the warmed blackbox model.

## Notes on reproducibility

- Set `CUDA_VISIBLE_DEVICES` outside the script if you want to select a GPU.
- The script sets random seeds for Python, NumPy, and PyTorch.
- Deterministic PyTorch settings are enabled by default, but some transformer/CUDA operations may still vary slightly across hardware and library versions.
- No data, credentials, or machine-specific paths are stored in the repository.

## Method summary

1. Load clinical notes and binary labels.
2. Load ICD-10 concepts and construct concept descriptions.
3. Warm up the pretrained encoder with a CLS-based blackbox classifier.
4. Encode ICD-10 concept descriptions with the warmed encoder.
5. Train the token-concept attention model:
   - token embeddings are compared with concept embeddings;
   - a NULL concept absorbs non-concept tokens;
   - Sparsemax produces sparse token-concept attention;
   - pooled concept relevance scores are used in `A_pool @ V @ O`;
   - group lasso and NULL/entropy regularizers follow the notebook.
6. Export predictive metrics and global concept rankings.
7. Optionally run perturbation analysis using the warmed blackbox model.
