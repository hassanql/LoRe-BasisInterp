# Phase 1 Gate Report

## Code
- Repository: `hassanql/LoRe-BasisInterp`
- Branch: `basis-reproducibility-test`
- Commit used for run: `032610aad0689d6f7203ee1b794d577cc1d1130a`
- Working tree clean before experiment run: yes

## Environment
- Provider: GCP
- Zone: `asia-southeast1-a`
- Machine: `a2-highgpu-1g`
- GPU: NVIDIA A100-SXM4-40GB
- Python: 3.10.12
- PyTorch: 2.3.0+cu121
- Transformers: 4.46.3

## Dataset
- Dataset: PRISM
- Split seed: `123`
- Train examples: `12999`
- Test examples: `13083`
- Seen examples: `20815`
- Unseen examples: `5267`
- Unique users: `1287`
- Unique dialogs: `7727`

## Embeddings
- Backbone: `Skywork/Skywork-Reward-Llama-3.1-8B-v0.2`
- Extraction: final-layer, last-token hidden state
- Dimension: `4096`
- Train embedding records: `12999`
- Test embedding records: `13083`
- Non-finite values: `0`
- Identical chosen/rejected pairs: `0`
- Validation: PASS

## LoRe Configuration
- Pairwise feature: `chosen_embedding - rejected_embedding`
- Rank sweep: `K in {5, 10, 20}`
- Rank seed: `42`
- Seed sweep at fixed `K=10`: `{0, 1, 2, 42}`
- Alpha: `1e4`
- Iterations: `20000`
- Learning rate: `0.5`

## LoRe Results

| Part | Run | K | Seed | Bases Kept | Basis FP | Train Acc | Test Acc | Test Std |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PART1-rank | K=5 | 5 | 42 | 4 | 2946.3696 | 0.8760 | 0.8770 | 0.1207 |
| PART1-rank | K=10 | 10 | 42 | 3 | 4746.9243 | 0.9258 | 0.9266 | 0.0997 |
| PART1-rank | K=20 | 20 | 42 | 5 | 6933.8164 | 0.9393 | 0.9314 | 0.1072 |
| PART2-seed | seed=0 | 10 | 0 | 4 | 4617.4072 | 0.8813 | 0.8798 | 0.1254 |
| PART2-seed | seed=1 | 10 | 1 | 4 | 4702.7339 | 0.9299 | 0.9263 | 0.1039 |
| PART2-seed | seed=2 | 10 | 2 | 5 | 4142.8623 | 0.9344 | 0.9327 | 0.1035 |
| PART2-seed | seed=42 | 10 | 42 | 3 | 4746.9243 | 0.9258 | 0.9266 | 0.0997 |

Canonical comparison row:

```text
PART2_K10_seed42: bases_kept=3, basis_fp=4746.9243, train_acc=0.9258, test_acc=0.9266, test_std=0.0997
```

## Artifact Package

Generated artifacts are intentionally not committed to Git. They are stored as
GitHub Release assets for this run.

SHA-256 hashes:

```text
7d9b9e7b752f103c328d08ac0ae6261af946c9df364102fe1e86c02975da3abb  phase1_pip_freeze.txt
3f930a28ec1ccb5c325e5696bb28322fb881476c11291521f08cf55c575516c8  phase1_run_log.txt
1caa7eb19c58a79469416348e519332b8eeed032dc0c53eb02da93f1ab8836e9  phase1_dataset_summary.txt
7599811dfdea75a01a805a3ff1a13c6511f3aa95d86d2bae15caa687de09946e  phase1_embedding_validation.txt
ff6923a065747954eb79ad8bbc371000dc9df658ef362ddb05d0dc29a76524a7  train_embeddings.pkl
3584986acd428204092e535df5946333a5319a8b5ec3fdf1793cdc8aee7cc625  test_embeddings.pkl
ec424d68e383042c19c7d7d68ad85390f11703a4517b8f206c28dccbb4b8db84  phase1_environment.txt
5bfe4db7f8679062b3d1e3ab55d39cc9ed345df78612e905fda96cb7414ff7c7  basis_matrices.pt
6771fad64d9f1108dbf607197cc833af79c89fc846a9d67cd9a99449c39f86f6  basis_reproducibility_results.csv
31dddb1c888dd6736113ff17edf74169cbd7e8e6008290e7fe7a7e6c268d3a35  basis_detail.csv
a111077012f992818d88019896317b470e71d9b6c5b55b8aa9d0b2fa0d54bbf8  phase1_nvidia_smi.txt
```

## Gate Status
- Gate 0 code version: PASS
- Gate 1 environment health: PASS
- Gate 2 dataset preparation: PASS
- Gate 3 embedding generation and validation: PASS
- Gate 4 LoRe reproducibility run: PASS
- Gate 5 team comparison: pending team reference outputs
