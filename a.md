# LoRe-BasisInterp — SAE Phase 2 Handoff Document

**Purpose:** Full technical write-up of everything done so far on Hassan’s SAE work, so another model (e.g. GPT) can continue without prior chat context.

**Date of this write-up:** 2026-07-10  
**Repo:** `hassanql/LoRe-BasisInterp`  
**Branch:** `hassan/sae-phase2`  
**Latest relevant commit:** `706de93` — *Add D3 audit: summary, feature-usage stats, LoRe-basis attribution*  
**Local monorepo path (Hassan’s machine):** `/Users/salih/Desktop/PRISM/LoRe-BasisInterp`  
**Local phase-2 artifact cache (not all committed):** `/Users/salih/Desktop/PRISM/phase2_artifacts/`

---

## 1. Who / scope / hard constraints

### Owner
- **Hassan** owns SAE training, evaluation, diagnostics, and basis–feature attribution only.

### Explicitly out of scope (do not do unless Hassan asks)
- Handoff packaging for Prerana/Ifesi beyond what already exists in scripts
- Semantic feature labeling / concept names
- LLM judge
- Persona vectors
- Behavioral validation
- Opening PRs (push branch only when asked)
- Claiming “feature X means sycophancy/verbosity/etc.”
- Treating decoder·V attribution as causal proof

### Infra constraints
- GCP VM: `prism-phase1-a100` (A100), zone `asia-southeast1-a`, project `hassanh-project`
- **Do not leave the A100 running.** Start only when needed; stop after scp/train.
- D3 is already trained; retrain only if necessary.

### Git policy
- Push only to `hassan/sae-phase2`
- Do **not** open a PR unless asked
- Do **not** commit large `.pt` / checkpoints / raw embeddings by default
- Small result CSVs/JSONs for D3 audit **were** committed under `sae/results/D3_16k_k256/`

---

## 2. Project goal (why SAE exists)

**LoRe** learns shared reward basis directions \(V\) across users. Each user has weights over those bases → personalized reward.

**Interpretation question:**  
> Do the learned LoRe bases correspond to human-interpretable preference concepts?

**SAE role (Method 1 — feature decomposition):**  
Train a sparse autoencoder on PRISM reward-model embeddings, then decompose each LoRe basis direction into sparse SAE features for later human/LLM inspection.

```text
PRISM embeddings (Skywork, 4096-d)
    → train TopK SAE
    → sparse features z
    → connect features to LoRe bases via decoder · V
    → (later, not done here) export top examples → labeling / judge
```

**Critical gate:** The SAE is only useful if it **preserves LoRe geometry**. If reconstruction destroys basis scores, feature labels may explain the SAE, not LoRe.

---

## 3. Phase 1 inputs (assumed available)

### Canonical LoRe basis
- File: `PRISM/basis_matrices.pt`
- Key: **`PART2_K10_seed42`**
- Matrix: `V` with shape **`[4096, 10]`** (10 bases stored)
- Basis \(j\) is **`V[:, j]`**
- User weights `W` shape **`[n_users, 10]`**
- **Operational kept bases:** only those with `max_user_weight_j >= 1e-2` (same rule as LoRe accuracy eval). For this run: **bases 1, 3, 9** (metadata `bases_kept = 3`). The other seven columns remain in `V` but have ~zero user mass.
- Note: **V columns are NOT unit-norm** (column norms ~1.4e3–1.6e3). This inflates raw `decoder · V` magnitudes; **ranking of features within a fixed basis is invariant** to that scale.

### Embeddings
- Skywork reward-model embeddings, dim **4096**
- PRISM train/test comparison embeddings (chosen + rejected per pair)
- Source scale: **26,082 comparisons × 2 = 52,164** response embeddings

### Phase 1 claim (careful wording)
> Approximately reproduced the LoRe basis directions and reward behavior.

Do **not** claim exact basis-matrix identity across machines without full matrix comparison.

---

## 4. SAE dataset (built and locked)

### Script
`sae/scripts/build_sae_dataset.py`

### Training data choice
Train on **individual response embeddings**:
```text
X = {all e_chosen} ∪ {all e_rejected}
```
**Not** on pairwise differences for training.  
Pairwise diffs `d = e_chosen − e_rejected` are used for **eval** (pair score correlation + LoRe preference accuracy).

### Split (seed 123)
| Split | Embeddings | Pairs |
|-------|-----------:|------:|
| train | 41,730 | 20,865 |
| val   | 5,216  | 2,608  |
| test  | 5,218  | 2,609  |
| **total** | **52,164** | **26,082** |

### Artifacts (local, usually gitignored)
```text
sae/data/sae_train.pt
sae/data/sae_val.pt
sae/data/sae_test.pt
sae/data/metadata.jsonl
sae/data/dataset_summary.json
```

### Metadata (for later text recovery)
Uses **`dialog_id`** (not `conversation_id`). Fields include embedding_id, source_split, pair_id, user_id, dialog_id, response_role (chosen/rejected), is_seen_user, original_index.

---

## 5. Model architecture (current code)

### Implementation
- `sae/src/topk_sae.py` — class `TopKSAE`
- Training: `sae/scripts/train_sae.py`
- Eval: `sae/scripts/evaluate_sae.py`
- Diagnostics: `sae/scripts/diagnose_sae.py`
- Attribution helpers: `sae/src/attribution.py`, `sae/scripts/analyze_basis_features.py`

### Architecture details
- Input dim: **4096**
- Encoder: `Linear(4096 → dict_size)` + bias
- Decoder: `Linear(dict_size → 4096)` + bias
- **Pre-encoder bias `b_pre`:** input centering; initialized from **train mean** when `center_inputs: true`
- Encode: `z = TopK(encoder(x − b_pre), k)` (or BatchTopK variant)
- Decode: `x_hat = decoder(z) + b_pre`
- **Unit-norm decoder columns** after each step when `normalize_decoder: true`
- Encoder init ≈ decoder transpose
- Optional **aux TopK loss** on dead features (revival): `aux_k_coef`, `aux_k`
- Optional **basis-score loss**: `MSE(x @ V, x_hat @ V)` with coef `basis_score_coef`  
  (**Warning:** `basis_score_coef=0.1` collapsed the model — see B4)

### Loss (successful recipe)
```text
L = recon_MSE(x_hat, x) + aux_k_coef * aux_MSE
```
Default successful knobs:
- `aux_k_coef = 0.03125`
- `normalize_decoder = true`
- `center_inputs = true`
- `basis_score_coef = 0.0` (no basis loss for champion)
- `batch_size = 256`, `lr = 3e-4`
- Early abort if dead rate > 0.9 at step 2000

---

## 6. Evaluation metrics & gates

### Reconstruction
- MSE, explained variance (EV)
- mean embedding / recon / error norms
- average active features

### LoRe preservation (individual embeddings)
For each basis \(j\):
```text
score_j(e)     = V[:, j] · e
score_j(e_hat) = V[:, j] · e_hat
```
Report Pearson (and Spearman) per basis; **mean** and **min** across bases.

### Pairwise
```text
d     = e_chosen − e_rejected
d_hat = e_hat_chosen − e_hat_rejected
```
- Pair score Pearson per basis  
- **LoRe pairwise preference accuracy** on original vs reconstructed  
- **Drop** = acc_original − acc_reconstructed  
  (negative drop = recon slightly *higher* acc — observed often)

### Feature health
- Live / dead counts and rates (train / val / test)
- Gini of activation frequencies (**all** features and **live-only**)
- Top 1% / 5% / 10% activation mass
- Effective number of features
- Active features per example (mean / median / p10 / p90)

### Provisional gates (used for “champion”)
| Gate | Threshold | D3 result |
|------|-----------|-----------|
| Mean basis Pearson | ≥ **0.95** | **0.9542 PASS** |
| Min basis Pearson | ≥ **0.90** | **0.9538 PASS** |
| LoRe accuracy drop | ≤ **0.02** (2 pp) | **−0.0086 PASS** |
| Dead-feature collapse | no severe collapse | train dead ~0.6%; test dead ~35% (OK-ish) |
| Gini | secondary / diagnostic | **high** (~0.91 live) — open issue |

### Model selection priority
1. LoRe basis-score preservation  
2. Pairwise LoRe accuracy preservation  
3. Reconstruction quality  
4. Dead feature rate  
5. Feature activation quality / inequality  
6. Interpretability of top examples (later)

**Do not select by MSE alone.**

---

## 7. Experiment history (full capacity program)

All improved runs use **center + unit-norm + aux** unless noted.  
Canonical eval: `PART2_K10_seed42`.  
Numbers below are **test** unless marked.

Local comparison tables live under:
- `/Users/salih/Desktop/PRISM/phase2_artifacts/all_comparison/master_comparison.csv`
- `/Users/salih/Desktop/PRISM/phase2_artifacts/sweep_d/d_comparison.csv`
- Per-run folders under `phase2_artifacts/sweep_*`

### 7.1 A0 — original large TopK (FAILED dead collapse)

| Field | Value |
|-------|------:|
| Config | 16384 / k64 TopK, **no** center/unit/aux |
| mean basis r | 0.8825 |
| min basis r | 0.8814 |
| EV / MSE | 0.981 / 0.0963 |
| lore_drop | −0.0268 |
| live test | **127** / dead **99.2%** |

**Lesson:** Large dict without unit-norm + aux + centering → almost all features die.

### 7.2 B0 — improved small baseline

| Field | Value |
|-------|------:|
| Config | 4096 / k32 + center + unit + aux |
| mean / min basis r | 0.9210 / 0.9202 |
| EV / MSE | 0.9869 / 0.0660 |
| lore_drop | −0.0235 |
| live test | 2806 (dead 31.5%) |
| gini_test_all / live | 0.858 / 0.793 |

Better health; still **fails** mean-basis ≥ 0.95 gate.

### 7.3 B1 — 8192 / k64 (main improved mid-size)

| Field | Value |
|-------|------:|
| mean / min basis r | 0.9364 / 0.9358 |
| EV / MSE | 0.9895 / 0.0532 |
| lore_drop | −0.0134 |
| live test | 4511 (dead 44.9%) |
| gini_test_all / live | 0.906 / 0.829 |

### 7.4 B3 — BatchTopK 8192 / k64

| Field | Value |
|-------|------:|
| mode | batch_topk |
| mean / min basis r | 0.9355 / 0.9350 |
| similar to B1; slightly different usage |

BatchTopK did **not** clearly beat TopK on LoRe gate.

### 7.5 B4 — basis-score loss coef 0.1 (**FAILED / collapsed**)

| Field | Value |
|-------|------:|
| basis_score_coef | **0.1** (too large) |
| mean basis r | **0.999** (misleading) |
| EV / MSE | **0.903 / 0.488** (terrible recon) |
| lore_drop | **+0.054** (acc *worse*) |
| live test | **63** (dead 99.2%) |
| avg active | ~4 (not 64) |

**Lesson:** Direct basis loss at 0.1 optimizes score correlation by collapsing the autoencoder. **Do not reuse 0.1.** If retrying basis loss, try much smaller coefs (not done yet).

### 7.6 C1 / C2 / C3 — large dict

| Run | dict/k | mean/min basis r | pair r | EV | dead_test | live_test | gini_all | lore_drop |
|-----|--------|------------------|--------|-----|-----------|-----------|----------|-----------|
| C1 | 16384/k64 TopK | 0.9355 / 0.9349 | 0.890 | 0.989 | 0.672 | 5382 | 0.955 | −0.008 |
| C2 | 16384/k64 BatchTopK | 0.9357 / 0.9351 | 0.887 | 0.989 | 0.633 | 6009 | 0.957 | −0.018 |
| C3 | 16384/k128 TopK | 0.9465 / 0.9461 | 0.901 | 0.991 | 0.537 | 7587 | 0.952 | −0.015 |

**Lesson:** Raising **k** helps LoRe more than just raising dict size. C3 closest so far but still **under** 0.95 mean.

### 7.7 D sweep — capacity on k (and one longer train)

| Run | Config | mean/min basis r | pair r | EV | MSE | dead_test | live_test | gini_all | gini_live | lore_drop |
|-----|--------|------------------|--------|-----|-----|-----------|-----------|----------|-----------|-----------|
| D1 | 16384/k128, **40k steps** | 0.9452 / 0.9447 | 0.904 | 0.991 | 0.0466 | 0.561 | 7187 | 0.957 | 0.902 | −0.0057 |
| D2 | 8192/k128 | 0.9475 / 0.9470 | 0.898 | 0.991 | 0.0455 | 0.241 | 6216 | 0.874 | 0.834 | −0.0062 |
| **D3** | **16384/k256** | **0.9542 / 0.9538** | **0.915** | **0.992** | **0.0384** | 0.349 | **10668** | 0.941 | **0.910** | **−0.0086** |
| D4 | 16384/k192 | 0.9489 / 0.9484 | 0.910 | 0.992 | 0.0415 | 0.443 | 9130 | 0.945 | 0.902 | −0.0115 |

Configs in repo:
- `sae/configs/exp_d1_16384_k128_40k.yaml`
- `sae/configs/exp_d2_8192_k128.yaml`
- `sae/configs/exp_d3_16384_k256.yaml`
- `sae/configs/exp_d4_16384_k192.yaml`

### 7.8 Champion decision

**Champion = D3**

```text
D3 = 16384 / k256 TopK + center + unit-norm + aux
     (basis_score_coef = 0)
```

Reasons:
1. **Only run that clearly passes mean basis r ≥ 0.95** (0.9542), with min also strong (0.9538)
2. Best EV / MSE among non-collapsed runs
3. Best pair score Pearson (~0.915)
4. LoRe accuracy preserved (even slightly higher on recon)
5. Train almost fully live (16285 / 16384; dead ~0.6%)
6. D4 (k192) and D1 (longer k128) did **not** beat D3 on mean basis r
7. D2 has **lower Gini** (better inequality) but fails the 0.95 LoRe gate (0.9475)

**Caveat:** D3 still has **high feature-usage inequality** (Gini live ~0.91). It is champion on **LoRe preservation**, not on “ideal uniform feature usage.”

---

## 8. D3 detailed metrics (reproduced audit)

### Config (resolved)
| Field | Value |
|-------|-------|
| dict_size | 16384 |
| k | 256 |
| sparsity | TopK |
| center_inputs | True (`b_pre` from train mean) |
| unit-norm decoder | True |
| aux_k_coef | 0.03125 |
| aux_k | 256 |
| basis_score_coef | 0.0 |
| max_steps | 20000 |
| batch / lr | 256 / 3e-4 |
| LoRe key | PART2_K10_seed42 |

### Main test metrics
| Metric | Value |
|--------|------:|
| Mean basis Pearson | **0.95418** |
| Min basis Pearson | **0.95376** |
| Mean pair Pearson | **0.91518** |
| Min pair Pearson | 0.90914 |
| Explained variance | 0.99241 |
| MSE | 0.03840 |
| LoRe acc original | 0.93008 |
| LoRe acc reconstructed | 0.93870 |
| LoRe accuracy drop | **−0.00862** |
| Avg active features | ~255.98 (≈ k) |
| Mean embedding norm | ~143.95 |
| Mean recon error norm | ~11.98 |

### Feature health
| Split | Live | Dead rate | Gini all | Gini live |
|-------|-----:|----------:|---------:|----------:|
| train | 16285 | 0.0060 | 0.9333 | 0.9329 |
| val | 10621 | 0.3517 | 0.9412 | 0.9093 |
| test | 10668 | 0.3489 | 0.9411 | **0.9096** |

### Feature-usage extras (test)
| Metric | Value |
|--------|------:|
| Top 1% activation mass | 0.410 |
| Top 5% activation mass | **0.856** |
| Top 10% activation mass | 0.948 |
| Effective # features | **~666** |
| Active/example mean / median | 255.98 / 256 |
| Active/example p10 / p90 | 256 / 256 |
| Active/example min / max | 230 / 256 |
| Max activation frequency | 0.956 |
| Median activation freq (live) | ~0.00077 |

**One-line takeaway:**  
> **D3 passes LoRe preservation but has high feature-usage inequality.**

---

## 9. D3 audit deliverables (Task completed 2026-07-09)

Committed under:

```text
sae/results/D3_16k_k256/
```

| File | Contents |
|------|----------|
| `D3_SUMMARY.md` | Human-readable champion summary |
| `feature_usage_stats.json` / `.csv` | Gini all/live, mass percentiles, effective N, active/example, live/dead, main metrics |
| `top_features_per_basis.csv` | Supplementary: all 10 bases; top 50 pos/neg **by contribution** |
| `top_features_per_basis_operational.csv` | **Primary:** bases **1, 3, 9** only; contribution-ranked |
| `BASIS_SCOPE.md` | Clarifies 10 stored vs 3 operational bases |
| `attribution_meta.json` | Shapes, formulas, keep mask, ranking notes |
| `sae_eval_summary.json` | Full eval dump |
| `sae_diagnostics_summary.json` | Diagnostics dump |
| `activation_stats.csv` | Per-feature activation stats |
| `active_per_example.json` | Active-count distribution |
| `basis_score_correlations.csv` | Per-basis correlations |
| `train_config_resolved.json` | Resolved train config |

Checkpoint (local, gitignored):
```text
sae/checkpoints/D3_16384_k256/model.pt   (~512MB after full scp)
```
Also mirrored under `phase2_artifacts/sweep_d/D3_16384_k256/` on Hassan’s machine.

### Attribution method (cleaned — contribution-ranked)

```text
alignment_ij           = decoder.weight[:, i] · V[:, j]
cosine_alignment_ij    = decoder.weight[:, i] · unit(V[:, j])
mean_abs_activation_i  = mean_e |z_i(e)|   over test (n=5218)
activation_frequency_i = mean_e 1[|z_i| > 0]
mean_abs_contribution  = mean_abs_activation_i * |alignment_ij|
```

CSV columns:
```text
basis_id, is_operational_kept, max_user_weight, rank, sign, feature_id,
alignment, cosine_alignment, mean_abs_activation, activation_frequency,
mean_abs_contribution
```

- Ranked by **mean_abs_contribution** (within sign of alignment)
- Primary file: operational bases **1, 3, 9** (300 rows)
- Supplementary file: all 10 bases (1000 rows)
- **No semantic names**; observational, not causal

#### Scale note
Decoder columns are unit-norm. V columns are large (~1.5e3).  
Cosine column is comparable across bases; raw alignment is not.  
Unit-normalizing V does **not** change within-basis feature order.

#### Empirical observation after cleanup
On operational bases, feature **2260** still often ranks #1 by contribution (high activation × alignment).  
Negative list now features with **real activation** (e.g. 8938, 15558) instead of near-dead alignment-only directions (old 8553).  
Shared high-contribution features across bases are still possible; do not over-interpret as “one feature = all bases.”
---

## 10. Code / config map

```text
sae/
  README.md                 # Phase 2 plan, gates, schema
  GCP_A100_RUNBOOK.md       # how to start/stop A100 and run jobs
  configs/
    topk_sae_baseline.yaml  # 4096/k32 improved
    topk_sae_large.yaml     # 16384/k64 template
    exp_b1_8192_k64.yaml
    exp_b3_batchtopk_8192.yaml
    exp_b4_basis_loss.yaml  # FAIL at coef 0.1
    exp_c1_16384_k64.yaml
    exp_c2_16384_batchtopk.yaml
    exp_c3_16384_k128.yaml
    exp_d1_16384_k128_40k.yaml
    exp_d2_8192_k128.yaml
    exp_d3_16384_k256.yaml  # CHAMPION
    exp_d4_16384_k192.yaml
  scripts/
    build_sae_dataset.py
    train_sae.py
    evaluate_sae.py
    diagnose_sae.py
    analyze_basis_features.py
    export_top_examples.py
    verify_sae_impl.py
    run_experiment.sh
    live_status.sh
  src/
    topk_sae.py
    metrics.py
    data.py
    io.py
    attribution.py
  results/D3_16k_k256/      # committed audit
  checkpoints/              # gitignored
  data/                     # gitignored
```

### Training environment
- Primarily **GCP A100** (`prism-phase1-a100`)
- Local Mac used for dataset prep, summary writing, attribution after scp of `model.pt`
- A100 should be **TERMINATED** when idle

---

## 11. What is done vs not done

### Done
- [x] Phase 2 plan + SAE scaffold
- [x] Dataset builder + metadata schema
- [x] TopK SAE training pipeline with center / unit-norm / aux
- [x] BatchTopK option
- [x] Optional basis-score loss (proved harmful at 0.1)
- [x] Eval: recon + LoRe basis r + pair r + LoRe accuracy drop
- [x] Diagnostics: live/dead, Gini, activation stats
- [x] Capacity sweeps B / C / D (including D4)
- [x] Select **D3** as LoRe-gate champion
- [x] D3 summary file
- [x] D3 feature-usage stats (JSON + CSV)
- [x] D3 top-50 pos/neg features per basis (numeric attribution only)
- [x] Commit + push audit to `hassan/sae-phase2` (`706de93`)
- [x] Confirm 10 stored vs 3 operational bases (1, 3, 9)
- [x] Regenerate attribution ranked by contribution + cosine + activation frequency
- [x] Primary operational CSV + supplementary all-10 CSV

### Explicitly **not** done (by design)
- [ ] Semantic naming of features or bases
- [ ] LLM judge labeling
- [ ] Persona vectors
- [ ] Behavioral validation
- [ ] Export of top **text** examples for labeling handoff (script exists; not run as part of D3 audit)
- [ ] Fixing high Gini / activation concentration
- [ ] Safer re-try of basis-score loss with tiny coef
- [ ] PR to main
---

## 12. Open scientific / engineering issues

1. **High Gini on D3**  
   Top 5% of features hold ~86% of activation mass; effective N ≈ 666 out of 16384. Dictionary capacity is underused.

2. **Train vs test live gap**  
   Train almost fully live; test ~35% dead. Some features may be train-only noise or rare modes.

3. **Shared high-contribution feature (2260)**  
   Still high on contribution ranking across operational bases (real activation, not a dead geometric artifact). Interpret carefully; not proof of a single shared concept.

4. **BatchTopK** did not win LoRe gate over TopK in this setup.

5. **Basis loss** needs much smaller coef or different formulation if revisited.

6. **Trade-off:** D2 has better Gini but fails mean basis ≥ 0.95. Improving inequality without losing LoRe gate is optional future SAE work — **not required** to freeze D3 as LoRe champion.

---

## 13. Suggested next steps (only if Hassan asks)

**Recommended status:** SAE training + technical attribution cleanup is in a good **stopping position**. Do **not** train more models / start A100 by default.

Optional later (only if asked):

1. Export top activating **text** examples for operational-basis top features (`export_top_examples.py`) — still no labels.
2. Gini-reduction experiments while holding LoRe gate — research, not required for freeze.
3. Tiny basis_score_coef sweep — optional.
4. Labeling / LLM judge / behavioral validation — **out of Hassan SAE-only scope** unless requested.

---

## 14. Quick answers for common questions

| Question | Answer |
|----------|--------|
| What is the champion SAE? | **D3: 16384/k256 TopK + center + unit-norm + aux** |
| Does it pass LoRe gates? | **Yes** (mean r 0.954, min r 0.954, acc drop −0.9 pp) |
| Live Gini (test)? | **≈ 0.910** |
| Gini all (test)? | **≈ 0.941** |
| Still champion? | **Yes for LoRe preservation**; inequality remains the flaw |
| Branch / commit? | `hassan/sae-phase2` (see latest attribution cleanup commit) |
| Where are audit files? | `sae/results/D3_16k_k256/` |
| Checkpoint path? | `sae/checkpoints/D3_16384_k256/model.pt` (local, **frozen**) |
| Canonical V? | `PART2_K10_seed42` in `basis_matrices.pt` |
| How many bases? | **10 stored; 3 operational (1, 3, 9)** |
| Primary attribution file? | `top_features_per_basis_operational.csv` (contribution-ranked) |
| Semantic labels? | **None assigned** |
| Causal claims? | **None** — attribution only |

---

## 15. Key commit timeline (git)

```text
cc958e3  Add SAE Phase 2 plan and scaffold
c1daf16  Implement SAE dataset builder
17547c6  Implement baseline TopK SAE training
9f6bb7c  Add LoRe accuracy preservation metric
ce29684  Add SAE activation diagnostics
8ee58e8  Improve TopK SAE training to reduce dead-feature collapse
1149b53  Add Gini metrics, BatchTopK, basis-score loss, and experiment configs
c5e73e0  Add C1–C3 large-dict SAE experiment configs
400481d  Add live_status.sh for one-line VM experiment progress
b0145d3  Add D1–D3 high-k capacity experiments and early-abort training
c594e8e  Add D4 16k/k192 config and SAE implementation verify script
706de93  Add D3 audit: summary, feature-usage stats, LoRe-basis attribution
```

---

## 16. Copy-paste status blurb

```text
SAE Phase 2 (Hassan): trained TopK SAEs on PRISM 4096-d chosen+rejected
embeddings with center + unit-norm decoder + aux dead-feature loss.
Swept B/C/D capacity. Champion is D3 = 16384/k256 TopK (20k steps).
D3 passes LoRe gates: mean basis r ≈ 0.954, min ≈ 0.954, pair r ≈ 0.915,
EV ≈ 0.992, LoRe acc drop ≈ −0.9pp. Feature usage remains unequal
(test Gini live ≈ 0.91; top 5% mass ≈ 86%; effective N ≈ 666).
Audit artifacts in sae/results/D3_16k_k256/. Attribution cleaned: ranked by
mean abs contribution; primary file for operational bases 1/3/9; cosine +
activation frequency included. 10 V columns stored, 3 operationally kept.
D3 checkpoint frozen. No labeling / judge / persona / PR / A100.
```

---

*End of handoff document.*
