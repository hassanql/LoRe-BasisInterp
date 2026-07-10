# D3 Summary — LoRe-preserving TopK SAE

## Config

| Field | Value |
|-------|-------|
| `dict_size` | 16384 |
| `k` | 256 |
| sparsity | **TopK** (`topk`) |
| center inputs (`b_pre` from train mean) | True |
| unit-norm decoder | True |
| aux dead-feature coef | 0.03125 |
| basis-score coef | 0.0 |
| max steps | 20000 |
| batch size / lr | 256 / 0.0003 |
| LoRe eval run | `PART2_K10_seed42` |
| train data | chosen + rejected 4096-d embeddings |

## Main metrics (test)

| Metric | Value | Gate |
|--------|------:|------|
| Mean basis Pearson | **0.9542** | ≥ 0.95 → **PASS** |
| Min basis Pearson | **0.9538** | ≥ 0.90 → **PASS** |
| Mean pair Pearson | **0.9152** | — |
| Explained variance | 0.9924 | — |
| MSE | 0.0384 | — |
| LoRe acc original | 0.9301 | — |
| LoRe acc reconstructed | 0.9387 | — |
| LoRe accuracy drop | **-0.0086** | ≤ 0.02 → **PASS** |

## Feature health

| Split | Live | Dead rate | Gini all | Gini live |
|-------|-----:|----------:|---------:|----------:|
| train | 16285 | 0.0060 | 0.9333 | 0.9329 |
| val | 10621 | 0.3517 | 0.9412 | 0.9093 |
| test | 10668 | 0.3489 | 0.9411 | 0.9096 |

### Test feature-usage extras

| Metric | Value |
|--------|------:|
| Top 1% activation mass | 0.4101 |
| Top 5% activation mass | 0.8557 |
| Top 10% activation mass | 0.9475 |
| Effective # features | 665.6 |
| Active/example mean | 255.98 |
| Active/example median | 256 |
| Active/example p10 / p90 | 256 / 256 |

## One-line takeaway

**D3 is LoRe-faithful but has concentrated feature usage.**

Live Gini (test) ≈ **0.910**; top 5% of features carry ≈ **86%** of activation mass.
D3 remains the frozen LoRe-preserving SAE champion; Gini is a secondary diagnostic, not a fail gate.

## LoRe basis scope (important)

Canonical run `PART2_K10_seed42` stores **all 10** columns of `V` (`[4096, 10]`).

**Operational kept bases** use the same rule as LoRe preference accuracy in `evaluate_sae.py`:

```text
kept if max_user_weight_j >= 1e-2
```

For this checkpoint of `W`:

| basis_id | max user weight | operational? |
|---------:|----------------:|:------------:|
| 0 | ~0 | no |
| **1** | **1.0** | **yes** |
| 2 | ~0 | no |
| **3** | **1.0** | **yes** |
| 4–8 | ~0 | no |
| **9** | **1.0** | **yes** |

So: **all 10 bases remain in the model matrix**; **3 bases (1, 3, 9) are operationally active** for personalized LoRe scores. Metadata field `bases_kept = 3` matches this.

- Primary attribution report: `top_features_per_basis_operational.csv` (bases 1, 3, 9)
- Supplementary (all 10): `top_features_per_basis.csv`

## Feature attribution (cleaned)

Ranked by **mean absolute contribution**, not raw alignment:

```text
alignment_ij          = decoder_i · V[:, j]
cosine_alignment_ij   = decoder_i · unit(V[:, j])   # decoder already unit-norm
mean_abs_activation_i = mean |z_i|
activation_frequency_i= mean 1[|z_i| > 0]
mean_abs_contribution = mean_abs_activation_i * |alignment_ij|
```

Within each basis, top 50 **positive** and top 50 **negative** features (by sign of alignment), sorted by contribution.

CSV columns: `basis_id, is_operational_kept, max_user_weight, rank, sign, feature_id, alignment, cosine_alignment, mean_abs_activation, activation_frequency, mean_abs_contribution`.

**Notes:**
- Attribution is observational (reward-lens style projection), **not causal**.
- No semantic feature names.
- Unit-normalizing `V` changes score scale across bases but **not** feature order within one basis.
- Feature **2260** still ranks high by contribution on operational bases (high activation × high alignment); inactive near-zero features no longer dominate the negative list.
