# LoRe basis scope for D3 attribution

## Question

Does “3 bases kept” mean only three bases exist, or that ten exist and three are operational?

## Answer

**All ten bases remain in `V`.**  
**Three bases are operationally active** for personalized LoRe predictions.

Canonical key: `PART2_K10_seed42` in `PRISM/basis_matrices.pt`.

```text
V.shape = [4096, 10]     # all columns stored
W.shape = [n_users, 10]  # user weights over all columns
bases_kept metadata = 3
```

### Keep rule (same as eval)

```python
kept_mask = user_w.max(dim=0).values >= 1e-2
```

This is the rule used in `sae/scripts/evaluate_sae.py` → `personalized_lore_accuracy`.

### Operational bases for PART2_K10_seed42

| basis_id | operational kept |
|---------:|:----------------:|
| 1 | yes |
| 3 | yes |
| 9 | yes |

All other basis ids (0, 2, 4, 5, 6, 7, 8) have max user weight ≈ 0 under this threshold.

### Reporting convention

| File | Role |
|------|------|
| `top_features_per_basis_operational.csv` | **Primary** — bases 1, 3, 9 only |
| `top_features_per_basis.csv` | **Supplementary** — all 10 bases |
| `attribution_meta.json` | formulas, keep mask, shapes |

### Ranking

Primary sort: **mean absolute contribution**  
(`mean|z_i| × |decoder_i · V[:, j]|`)

Keep raw alignment and cosine alignment as columns.  
Do not treat attribution as causal.
