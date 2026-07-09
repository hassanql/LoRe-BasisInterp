# Phase 2: SAE Training for LoRe Basis Interpretation

## Goal

Train a Sparse Autoencoder (SAE) on PRISM reward-model embeddings and use it to
decompose LoRe basis directions into sparse, inspectable features.

The SAE is only useful for this project if it preserves the reward-relevant
geometry of LoRe. Therefore, model selection is based not only on reconstruction
quality, but also on LoRe basis-score preservation and pairwise preference
accuracy preservation.

This phase should produce a clean, reproducible SAE pipeline that can support
the team's later labeling and validation work.

## Project Context

LoRe learns shared reward basis directions across users. Each user has weights
over these shared bases, giving a personalized reward function.

Our interpretation question is:

> Do the learned LoRe bases correspond to human-interpretable preference concepts?

This SAE pipeline supports Method 1: feature decomposition labeling.

Expected flow:

```text
PRISM embeddings
    -> train SAE
    -> sparse features
    -> connect features to LoRe bases
    -> export top features/examples
    -> LLM judge and behavioral validation
```

## Non-Goals

This phase does not assign final concept labels.

Do not move to labeling until the SAE passes the LoRe-preservation gate.

Do not use this phase to:

- change the PRISM split;
- change the Skywork embedding extraction;
- retrain the reward model;
- tune LoRe itself;
- generate final labels for bases;
- run the LLM judge;
- claim causal feature meaning from SAE attribution alone.

## Inputs From Phase 1

Required generated artifacts:

```text
PRISM/data/prism/train_embeddings.pkl
PRISM/data/prism/test_embeddings.pkl
PRISM/basis_matrices.pt
```

These are local/generated artifacts and should not be committed to Git.

The Phase 1 reproducibility result should be described as:

> Approximately reproduced the LoRe basis directions and reward behavior.

Do not claim exact basis-matrix identity across machines unless full matrix
comparison confirms it.

## Canonical LoRe Basis

For the first SAE evaluation, use the canonical run:

```text
PART2_K10_seed42
```

from:

```text
basis_matrices.pt
```

Do not accidentally evaluate against all exploratory runs or the wrong value of
`K`.

If the team later changes the canonical seed or rank, update this README before
rerunning.

## Notation

Each PRISM comparison has:

```text
e_chosen   = chosen response embedding, shape [4096]
e_rejected = rejected response embedding, shape [4096]
d_pair     = e_chosen - e_rejected
```

The LoRe basis matrix is stored in code as:

```text
V
```

If:

```text
V.shape == [4096, K]
```

then basis `j` is:

```text
V[:, j]
```

If using row-wise notation in reports, define:

```text
A = V.T
```

so:

```text
A.shape == [K, 4096]
```

Be explicit about this to avoid confusion.

## SAE Training Data

The first SAE should be trained on individual response embeddings:

```text
X = {all e_chosen} union {all e_rejected}
```

For the current PRISM artifacts:

```text
26,082 comparisons x 2 = 52,164 response embeddings
```

Pairwise difference vectors are not the first training data. They are used for
evaluation because LoRe preference accuracy is computed on chosen-minus-rejected
differences.

## Metadata Schema

Every embedding in the SAE dataset must preserve enough metadata for Prerana and
Ifesi to recover the original text examples.

Use `dialog_id`, not `conversation_id`, because this is the metadata key in the
current PRISM artifacts.

Required metadata fields:

```json
{
  "embedding_id": "unique integer or string ID",
  "source_split": "train or test",
  "pair_id": "comparison-level ID if available",
  "user_id": "PRISM user ID",
  "dialog_id": "PRISM dialog ID",
  "response_role": "chosen or rejected",
  "is_seen_user": true,
  "original_index": 0
}
```

If a field is missing in the artifact, record it as `null` rather than silently
dropping the column.

## Output Dataset Artifacts

The dataset-building script should produce:

```text
sae/data/sae_train.pt
sae/data/sae_val.pt
sae/data/sae_test.pt
sae/data/metadata.jsonl
sae/data/dataset_summary.json
```

The `.pt` files should contain tensors only.

The `metadata.jsonl` file should contain one JSON object per embedding, aligned
by index with the tensor rows.

Large `.pt`, `.pkl`, and `.jsonl` files should not be committed unless the team
explicitly agrees.

## Baseline SAE

Start with a simple TopK SAE, then use the improved training recipe if dead
features collapse.

### Improved default (`sae/configs/topk_sae_baseline.yaml`)

```yaml
input_dim: 4096
architecture: topk_sae
dict_size: 4096
k: 32
loss: reconstruction_mse_plus_aux_dead
train_data: chosen_plus_rejected_embeddings
canonical_lore_run_key: PART2_K10_seed42
```

Training knobs included in this config:

- unit-norm decoder columns after each step;
- input centering via pre-encoder bias initialized from the train mean;
- auxiliary TopK reconstruction on dead features (`aux_k_coef`);
- longer default schedule (`max_steps: 20000`).

### Large-dictionary control (`sae/configs/topk_sae_large.yaml`)

```yaml
dict_size: 16384
k: 64
```

Use this after dead-feature rate is under control. The original Phase 2 full-A100
run used 16384/64 without aux/unit-norm and collapsed to ~130 live features.

This is still a baseline family, not the final claim.

See also `sae/GCP_A100_RUNBOOK.md` for VM start/stop and train commands.

## Script Responsibilities

### `build_sae_dataset.py`

Convert Phase 1 embedding artifacts into clean SAE train/validation/test tensors
with metadata.

Required behavior:

- load `train_embeddings.pkl` and `test_embeddings.pkl`;
- extract chosen and rejected embeddings;
- confirm all embeddings are finite and 4096-dimensional;
- preserve `user_id`, `dialog_id`, split, and chosen/rejected status;
- create train/validation/test splits;
- save tensor files and metadata files;
- write a dataset summary.

Acceptance criteria:

- all embeddings have shape `[4096]`;
- no NaN or Inf values;
- metadata rows align exactly with tensor rows;
- `dialog_id` is preserved;
- chosen/rejected status is preserved;
- dataset summary reports the number of examples in each split.

### `train_sae.py`

Train a baseline TopK SAE on individual response embeddings.

Required outputs:

```text
sae/checkpoints/topk_sae_baseline.pt
sae/results/train_log.csv
sae/results/train_config_resolved.yaml
```

Acceptance criteria:

- training runs end-to-end;
- training loss decreases;
- validation loss is computed;
- checkpoint can be loaded by `evaluate_sae.py`;
- the resolved config is saved.

### `evaluate_sae.py`

Evaluate whether the SAE is good enough for LoRe interpretation.

Standard reconstruction metrics:

- reconstruction MSE;
- explained variance;
- mean embedding norm;
- mean reconstruction norm;
- reconstruction error norm;
- average active features;
- dead feature rate;
- activation frequency distribution.

LoRe preservation metrics on individual embeddings:

```text
original_score_j      = dot(V[:, j], e)
reconstructed_score_j = dot(V[:, j], e_hat)
```

Report:

- Pearson correlation per basis;
- Spearman correlation per basis;
- mean absolute score error per basis;
- relative score error per basis.

LoRe preservation metrics on pairwise differences:

```text
d_pair     = e_chosen - e_rejected
d_pair_hat = e_hat_chosen - e_hat_rejected
```

For each basis `j`, compare:

```text
dot(V[:, j], d_pair)
dot(V[:, j], d_pair_hat)
```

Also compare LoRe pairwise preference accuracy using original embeddings versus
reconstructed embeddings.

Initial acceptance rule:

```text
mean basis-score correlation >= 0.95
minimum basis-score correlation >= 0.90
LoRe pairwise accuracy drop <= 1-2 percentage points
no severe dead-feature collapse
```

These thresholds are provisional until baseline results are available.

Required outputs:

```text
sae/results/sae_eval_summary.csv
sae/results/basis_score_correlations.csv
sae/results/pairwise_accuracy_reconstruction.csv
sae/results/activation_stats.csv
```

Small final CSVs can be committed if the team agrees. Raw activations should not
be committed.

### `diagnose_sae.py`

Inspect feature usage before any labeling work.

Required outputs:

```text
sae/results/sae_diagnostics_summary.json
sae/results/top_active_features.csv
```

Diagnostics should report:

- live and dead feature counts per split;
- top active features per split;
- activation frequency and mean active value;
- decoder feature norm statistics.

### `analyze_basis_features.py`

Estimate which SAE features contribute most to each LoRe basis.

For feature `i` and basis `j`, use the approximate attribution:

```text
contribution_ij(e) = z_i(e) * dot(decoder_i, V[:, j])
```

where:

```text
z_i(e)      = SAE feature activation for embedding e
decoder_i  = SAE decoder direction for feature i
V[:, j]    = LoRe basis j
```

This is an attribution heuristic, not proof of causal feature meaning.

Aggregate statistics:

- mean signed contribution;
- mean absolute contribution;
- activation frequency;
- decoder-basis alignment;
- average activation when active;
- positive/negative contribution direction.

Required outputs:

```text
sae/results/top_features_per_basis.csv
sae/results/basis_feature_contributions.csv
sae/results/feature_activation_stats.csv
```

### `export_top_examples.py`

Export top activating examples for each important SAE feature so Prerana can map
them back to text and Ifesi can run LLM-judge labeling.

Required outputs:

```text
sae/results/top_examples_per_feature.csv
sae/results/top_examples_per_basis.md
```

Each row should include:

- feature ID;
- basis ID if relevant;
- embedding ID;
- activation value;
- contribution value;
- user ID;
- dialog ID;
- chosen/rejected status;
- source split;
- original index.

Large top-example dumps should stay ignored unless the team explicitly wants
them committed.

## Model Selection

Select SAE checkpoints using this priority order:

1. LoRe basis-score preservation;
2. pairwise LoRe accuracy preservation;
3. reconstruction quality;
4. dead feature rate;
5. feature activation quality;
6. interpretability of top examples.

Do not select only by reconstruction MSE.

## Hyperparameter Sweep

After the first baseline works, run a small controlled sweep:

| Run | dict_size | k | Purpose |
| --- | ---: | ---: | --- |
| A | 8192 | 32 | smaller/faster baseline |
| B | 16384 | 64 | main baseline |
| C | 32768 | 64 | larger dictionary |
| D | 16384 | 32 | more sparse |
| E | 16384 | 128 | less sparse |

Do not run a large sweep until the dataset schema and evaluation script are
locked.

## Git Tracking Policy

Do commit:

```text
sae/README.md
sae/configs/*.yaml
sae/scripts/*.py
sae/src/*.py
sae/results/.gitkeep
small final CSV summaries if the team agrees
```

Do not commit by default:

```text
*.pkl
*.pt
*.pth
*.ckpt
wandb/
checkpoints/
sae/data/
raw activations
large top-example dumps
large intermediate CSVs
```

Recommended `.gitignore` additions:

```gitignore
# SAE generated artifacts
sae/data/
sae/checkpoints/
sae/wandb/
sae/results/*.pt
sae/results/*.pkl
sae/results/*activations*
sae/results/*top_examples*
*.ckpt
*.pth
```

## Phase 2 Definition Of Done

Phase 2 SAE setup is complete when:

- the README and artifact schema are merged or approved;
- the SAE dataset builder works and preserves metadata;
- the baseline TopK SAE trains end-to-end;
- evaluation reports reconstruction and LoRe-preservation metrics;
- the provisional LoRe-preservation gate is applied;
- top basis-feature contribution tables are produced;
- top activating embedding IDs are exported;
- Prerana can map examples back to conversations/dialogs;
- Ifesi can use the exported examples for LLM-judge labeling.

## Team Handoff Summary

When reporting progress, use this structure:

```text
I completed the SAE dataset builder and baseline TopK SAE training.
The model was trained on chosen + rejected 4096-dim embeddings.
I evaluated reconstruction quality and LoRe preservation against PART2_K10_seed42.
The current model [passes/fails] the provisional gate.
I exported top feature IDs and embedding IDs for Prerana/Ifesi.
Next I will [improve SAE / run a small sweep / prepare labeling inputs].
```

## Key Warning

If the SAE does not preserve LoRe basis scores, then feature labels may explain
the SAE reconstruction rather than the original LoRe reward basis.

Labeling should start only after the LoRe-preservation gate passes or the team
explicitly decides to proceed with a clearly marked failed-gate baseline.
