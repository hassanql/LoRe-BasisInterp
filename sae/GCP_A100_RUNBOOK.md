# GCP A100 Runbook (PRISM / SAE)

Machine used for Phase 1/2:

| Field | Value |
|-------|-------|
| Project | `hassanh-project` |
| Account | `hassanh@aims.ac.za` |
| Instance | `prism-phase1-a100` |
| Zone | `asia-southeast1-a` |
| Type | `a2-highgpu-1g` (NVIDIA A100 40GB) |
| Status when idle | **TERMINATED** (stop after work) |

Local machine already has `gcloud` configured and SSH key `~/.ssh/google_compute_engine`.

---

## Cost rule

- **Start** the instance only when you are about to train/eval.
- **Stop** as soon as jobs finish.
- Do not leave the A100 running overnight unless intentionally.

---

## 1. Start the instance

```bash
gcloud compute instances start prism-phase1-a100 \
  --zone=asia-southeast1-a \
  --project=hassanh-project
```

Wait until `STATUS` is `RUNNING`:

```bash
gcloud compute instances list --project=hassanh-project
```

---

## 2. SSH in

```bash
gcloud compute ssh prism-phase1-a100 \
  --zone=asia-southeast1-a \
  --project=hassanh-project
```

Optional: open a remote shell from a local command with a working directory set after login.

---

## 3. One-time / occasional setup on the VM

These paths may already exist from Phase 1. Adjust if your home directory layout differs.

```bash
# GPU check
nvidia-smi

# Python env (example; use whatever already exists on the box)
cd ~
# Prefer reusing the Phase 1 venv if present:
#   source ~/LoRe-BasisInterp/.venv/bin/activate
# or create one:
#   python3 -m venv ~/venvs/lore
#   source ~/venvs/lore/bin/activate
#   pip install -r requirements.txt

# Clone / update the fork branch
# (if repo already exists, just fetch + checkout)
cd ~
if [ ! -d LoRe-BasisInterp ]; then
  git clone https://github.com/hassanql/LoRe-BasisInterp.git
fi
cd LoRe-BasisInterp
git fetch origin
git checkout hassan/sae-phase2
git pull origin hassan/sae-phase2
```

### Phase 1 artifacts the SAE needs

On the VM you need:

```text
PRISM/data/prism/train_embeddings.pkl
PRISM/data/prism/test_embeddings.pkl
PRISM/basis_matrices.pt
```

If they live only on the local laptop under `~/Desktop/PRISM/phase1_artifacts/`, copy them up:

```bash
# from local machine (laptop), while instance is RUNNING:
gcloud compute scp \
  --zone=asia-southeast1-a \
  --project=hassanh-project \
  ~/Desktop/PRISM/phase1_artifacts/train_embeddings.pkl \
  ~/Desktop/PRISM/phase1_artifacts/test_embeddings.pkl \
  ~/Desktop/PRISM/phase1_artifacts/basis_matrices.pt \
  prism-phase1-a100:~/LoRe-BasisInterp/PRISM/data/prism/

# basis_matrices.pt is expected at PRISM/basis_matrices.pt by configs:
gcloud compute ssh prism-phase1-a100 \
  --zone=asia-southeast1-a \
  --project=hassanh-project \
  --command='mkdir -p ~/LoRe-BasisInterp/PRISM/data/prism && mv ~/LoRe-BasisInterp/PRISM/data/prism/basis_matrices.pt ~/LoRe-BasisInterp/PRISM/basis_matrices.pt 2>/dev/null || true'
```

If Phase 1 left artifacts already on the VM disk, skip the scp and just confirm paths:

```bash
ls -lh ~/LoRe-BasisInterp/PRISM/data/prism/*.pkl
ls -lh ~/LoRe-BasisInterp/PRISM/basis_matrices.pt
```

---

## 4. Build SAE dataset (once per embedding version)

```bash
cd ~/LoRe-BasisInterp
source .venv/bin/activate   # or your env

python sae/scripts/build_sae_dataset.py \
  --train-embeddings PRISM/data/prism/train_embeddings.pkl \
  --test-embeddings PRISM/data/prism/test_embeddings.pkl \
  --output-dir sae/data
```

---

## 5. Train improved TopK SAE

Default improved config: `dict_size=4096`, `k=32`, unit-norm decoder, aux dead loss, 20k steps.

```bash
python sae/scripts/train_sae.py \
  --config sae/configs/topk_sae_baseline.yaml \
  --device cuda
```

Larger dictionary control:

```bash
python sae/scripts/train_sae.py \
  --config sae/configs/topk_sae_large.yaml \
  --device cuda
```

Ad-hoc overrides:

```bash
python sae/scripts/train_sae.py \
  --config sae/configs/topk_sae_baseline.yaml \
  --dict-size 8192 --k 64 --max-steps 20000 \
  --device cuda
```

---

## 6. Evaluate + diagnose

```bash
python sae/scripts/evaluate_sae.py \
  --checkpoint sae/checkpoints/topk_sae_baseline.pt \
  --data-dir sae/data \
  --basis-matrices PRISM/basis_matrices.pt \
  --run-key PART2_K10_seed42 \
  --split test \
  --device cuda

python sae/scripts/diagnose_sae.py \
  --checkpoint sae/checkpoints/topk_sae_baseline.pt \
  --data-dir sae/data \
  --device cuda
```

### Gate (provisional)

```text
mean basis-score correlation >= 0.95
min  basis-score correlation >= 0.90
LoRe pairwise accuracy drop  <= 0.02
no severe dead-feature collapse
```

---

## 7. Pull small results back to laptop (optional)

```bash
# from local machine
mkdir -p ~/Desktop/PRISM/phase2_artifacts/improved_run
gcloud compute scp --recurse \
  --zone=asia-southeast1-a \
  --project=hassanh-project \
  prism-phase1-a100:~/LoRe-BasisInterp/sae/results \
  ~/Desktop/PRISM/phase2_artifacts/improved_run/
```

Do **not** commit large `.pt` checkpoints unless the team explicitly wants that.

---

## 8. Stop the instance (mandatory when done)

```bash
gcloud compute instances stop prism-phase1-a100 \
  --zone=asia-southeast1-a \
  --project=hassanh-project
```

Confirm:

```bash
gcloud compute instances list --project=hassanh-project
# STATUS should be TERMINATED
```

---

## Agent / teammate push policy

- Push code only to branch: **`hassan/sae-phase2`**
- **Never** open a PR unless Hassan asks
- Large artifacts (`.pkl`, `.pt`, embeddings) stay on disk / scp, not in git

---

## Quick command cheat sheet

```bash
# status
gcloud compute instances list --project=hassanh-project

# start
gcloud compute instances start prism-phase1-a100 --zone=asia-southeast1-a --project=hassanh-project

# ssh
gcloud compute ssh prism-phase1-a100 --zone=asia-southeast1-a --project=hassanh-project

# stop
gcloud compute instances stop prism-phase1-a100 --zone=asia-southeast1-a --project=hassanh-project
```
