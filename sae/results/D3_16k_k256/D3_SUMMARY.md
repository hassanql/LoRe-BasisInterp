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

**D3 passes LoRe preservation but has high feature-usage inequality.**

Live Gini (test) ≈ **0.910**; top 5% of features carry ≈ **86%** of activation mass.
