#!/usr/bin/env python3
"""Sanity-check TopK SAE implementation and eval wiring (no training)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(REPO_ROOT))

from sae.src.metrics import pearson_corr_by_column, reconstruction_mse  # noqa: E402
from sae.src.topk_sae import TopKSAE  # noqa: E402


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    ok = True

    # --- unit tests on synthetic data ---
    m = TopKSAE(64, 128, k=8, sparsity_mode="topk").to(device)
    x = torch.randn(32, 64, device=device)
    m.set_pre_bias(x.mean(0))
    # decoder unit norm after normalize (no autograd live yet)
    m.normalize_decoder_()
    norms = torch.linalg.norm(m.decoder.weight, dim=0)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5):
        print("FAIL: decoder columns not unit norm", norms[:5]); ok = False
    else:
        print("OK: decoder unit-norm columns")

    x_hat, z = m(x)
    active = (z != 0).sum(-1)
    if not (active <= 8).all():
        print("FAIL: topk produced >k actives"); ok = False
    else:
        print("OK: topk active count <= k")

    # encode/decode uses b_pre consistently: x_hat = D(z) + b_pre, z=enc(x-b_pre)
    pre = m.encode_pre_acts(x)
    z2 = m._topk_activate(pre, m.k)
    x_hat2 = m.decoder(z2) + m.b_pre
    if not torch.allclose(x_hat, x_hat2, atol=1e-5):
        print("FAIL: forward != manual encode/decode"); ok = False
    else:
        print("OK: forward matches encode/decode + b_pre")

    # gradient flows (must not inplace-modify params between forward and backward)
    loss = F.mse_loss(x_hat, x)
    loss.backward()
    if m.encoder.weight.grad is None or m.decoder.weight.grad is None:
        print("FAIL: missing grads"); ok = False
    else:
        print("OK: grads flow to encoder/decoder")

    # training-order check: backward -> step -> normalize (as in train_sae.py)
    m2 = TopKSAE(64, 128, k=8, sparsity_mode="topk").to(device)
    opt = torch.optim.Adam(m2.parameters(), lr=1e-3)
    opt.zero_grad(set_to_none=True)
    xh2, _ = m2(x.detach())
    F.mse_loss(xh2, x.detach()).backward()
    opt.step()
    m2.normalize_decoder_()
    print("OK: train order backward->step->normalize_decoder works")

    # BatchTopK total budget
    mb = TopKSAE(64, 128, k=8, sparsity_mode="batch_topk").to(device)
    zb = mb.encode(x)
    total_active = int((zb != 0).sum().item())
    if total_active > 32 * 8:
        print("FAIL: batch topk total actives", total_active); ok = False
    else:
        print(f"OK: batch_topk total actives={total_active} <= {32*8}")

    # --- optional real-data checks ---
    data_dir = Path("sae/data")
    basis_path = Path("PRISM/basis_matrices.pt")
    if data_dir.exists() and (data_dir / "sae_test.pt").exists():
        xt = torch.load(data_dir / "sae_test.pt", map_location="cpu").float()[:256]
        print(f"OK: loaded test embeddings {tuple(xt.shape)} finite={bool(torch.isfinite(xt).all())}")
        if xt.shape[1] != 4096:
            print("FAIL: expected dim 4096"); ok = False
        # pair with a real checkpoint if present
        for ckpt_path in [
            Path("sae/checkpoints/D3_16384_k256/model.pt"),
            Path("sae/checkpoints/D2_8192_k128/model.pt"),
        ]:
            if not ckpt_path.exists():
                continue
            ckpt = torch.load(ckpt_path, map_location="cpu")
            cfg = ckpt["config"]
            t = cfg.get("training", {})
            model = TopKSAE(
                int(cfg["input_dim"]),
                int(cfg["dict_size"]),
                int(cfg["k"]),
                normalize_decoder=bool(t.get("normalize_decoder", True)),
                aux_k=int(t.get("aux_k", cfg["k"])),
                sparsity_mode=str(t.get("sparsity_mode", "topk")),
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device).eval()
            with torch.no_grad():
                xh, zz = model(xt.to(device))
                mse = float(reconstruction_mse(xt.to(device), xh).item())
                act = (zz != 0).float().sum(-1)
            print(
                f"OK: {ckpt_path.name} recon_mse={mse:.4f} "
                f"active mean={float(act.mean()):.2f} k={model.k} "
                f"decoder_norm_mean={float(torch.linalg.norm(model.decoder.weight, dim=0).mean()):.4f}"
            )
            if mse > 1.0:
                print("WARN: unusually high MSE on sample")
            if basis_path.exists():
                V = torch.load(basis_path, map_location="cpu")["PART2_K10_seed42"]["V"].float()
                print(f"OK: basis V shape={tuple(V.shape)} (expect [4096, K])")
                if V.shape[0] != 4096:
                    print("FAIL: V first dim not 4096"); ok = False
                with torch.no_grad():
                    s0 = xt @ V
                    s1 = xh.cpu() @ V
                    pr = pearson_corr_by_column(s0, s1)
                print(f"OK: sample mean basis pearson={float(pr.mean()):.4f} (all K={V.shape[1]} bases)")
            break
    else:
        print("SKIP: sae/data not present (synthetic checks only)")

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
