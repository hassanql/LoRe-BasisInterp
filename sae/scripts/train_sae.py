#!/usr/bin/env python3
"""Train the baseline / improved TopK SAE."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
SAE_DIR = SCRIPT_DIR.parent
REPO_ROOT = SAE_DIR.parent
sys.path.append(str(REPO_ROOT))

from sae.src.io import ensure_dir, read_simple_yaml, write_json  # noqa: E402
from sae.src.metrics import explained_variance, reconstruction_mse  # noqa: E402
from sae.src.topk_sae import TopKSAE  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="sae/configs/topk_sae_baseline.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--checkpoint-dir", default="sae/checkpoints")
    parser.add_argument("--results-dir", default="sae/results")
    parser.add_argument("--dict-size", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--aux-k-coef", type=float, default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> dict:
    config = read_simple_yaml(args.config)
    data_cfg = dict(config.get("data", {}))
    train_cfg = dict(config.get("training", {}))
    if args.data_dir is not None:
        data_cfg["output_dir"] = args.data_dir
    if args.dict_size is not None:
        config["dict_size"] = args.dict_size
    if args.k is not None:
        config["k"] = args.k
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        train_cfg["learning_rate"] = args.learning_rate
    if args.max_steps is not None:
        train_cfg["max_steps"] = args.max_steps
    if args.aux_k_coef is not None:
        train_cfg["aux_k_coef"] = args.aux_k_coef
    # Defaults for improved training knobs (backward compatible if missing).
    train_cfg.setdefault("aux_k_coef", 0.03125)
    train_cfg.setdefault("dead_feature_threshold", 1.0e-5)
    train_cfg.setdefault("normalize_decoder", True)
    train_cfg.setdefault("center_inputs", True)
    train_cfg.setdefault("activation_ema_beta", 0.99)
    train_cfg.setdefault("aux_k", train_cfg.get("k", config.get("k", 64)))
    config["data"] = data_cfg
    config["training"] = train_cfg
    return config


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    args = parse_args()
    config = resolve_config(args)
    data_dir = Path(config["data"]["output_dir"])
    checkpoint_dir = ensure_dir(args.checkpoint_dir)
    results_dir = ensure_dir(args.results_dir)
    device = choose_device(args.device)
    train_cfg = config["training"]

    train_x = torch.load(data_dir / "sae_train.pt", map_location="cpu").float()
    val_x = torch.load(data_dir / "sae_val.pt", map_location="cpu").float()
    train_loader = DataLoader(
        TensorDataset(train_x),
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        drop_last=True,
    )

    model = TopKSAE(
        input_dim=int(config["input_dim"]),
        dict_size=int(config["dict_size"]),
        k=int(config["k"]),
        normalize_decoder=bool(train_cfg["normalize_decoder"]),
        aux_k=int(train_cfg.get("aux_k", config["k"])),
    ).to(device)

    if bool(train_cfg["center_inputs"]):
        train_mean = train_x.mean(dim=0)
        model.set_pre_bias(train_mean.to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg["learning_rate"]))

    max_steps = int(train_cfg["max_steps"])
    log_every = int(train_cfg.get("log_every", 100))
    eval_every = int(train_cfg.get("eval_every", 500))
    checkpoint_every = int(train_cfg.get("checkpoint_every", 1000))
    aux_k_coef = float(train_cfg["aux_k_coef"])
    dead_threshold = float(train_cfg["dead_feature_threshold"])
    ema_beta = float(train_cfg["activation_ema_beta"])

    # EMA of fraction of examples where each feature is active.
    act_freq_ema = torch.zeros(model.dict_size, device=device)

    log_path = results_dir / "train_log.csv"
    fieldnames = [
        "step",
        "train_mse",
        "train_aux_mse",
        "train_loss",
        "dead_feature_rate",
        "live_features",
        "val_mse",
        "val_explained_variance",
    ]
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        step = 0
        while step < max_steps:
            for (batch,) in train_loader:
                step += 1
                batch = batch.to(device)
                dead_mask = act_freq_ema < dead_threshold

                optimizer.zero_grad(set_to_none=True)
                out = model.forward_with_loss(
                    batch,
                    dead_mask=dead_mask,
                    aux_k_coef=aux_k_coef,
                )
                out["loss"].backward()
                optimizer.step()
                model.normalize_decoder_()

                with torch.no_grad():
                    batch_freq = (out["z"] != 0).float().mean(dim=0)
                    act_freq_ema.mul_(ema_beta).add_(batch_freq, alpha=1.0 - ema_beta)
                    dead_rate = float((act_freq_ema < dead_threshold).float().mean().item())
                    live_features = int((act_freq_ema >= dead_threshold).sum().item())

                should_eval = step == 1 or step % eval_every == 0 or step == max_steps
                should_log = step == 1 or step % log_every == 0 or should_eval
                val_mse = ""
                val_ev = ""
                if should_eval:
                    with torch.no_grad():
                        n_val = min(len(val_x), int(train_cfg["batch_size"]) * 8)
                        val_batch = val_x[:n_val].to(device)
                        val_hat, _ = model(val_batch)
                        val_mse = float(reconstruction_mse(val_batch, val_hat).item())
                        val_ev = float(explained_variance(val_batch, val_hat).item())
                if should_log:
                    writer.writerow(
                        {
                            "step": step,
                            "train_mse": float(out["recon_mse"].item()),
                            "train_aux_mse": float(out["aux_mse"].item()),
                            "train_loss": float(out["loss"].item()),
                            "dead_feature_rate": dead_rate,
                            "live_features": live_features,
                            "val_mse": val_mse,
                            "val_explained_variance": val_ev,
                        }
                    )
                    f.flush()
                    if should_eval:
                        print(
                            f"step={step} recon_mse={float(out['recon_mse'].item()):.5f} "
                            f"aux_mse={float(out['aux_mse'].item()):.5f} "
                            f"dead={dead_rate:.4f} live={live_features} "
                            f"val_mse={val_mse} val_ev={val_ev}"
                        )
                if step % checkpoint_every == 0 or step == max_steps:
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "config": config,
                            "step": step,
                            "act_freq_ema": act_freq_ema.detach().cpu(),
                        },
                        checkpoint_dir / "topk_sae_baseline.pt",
                    )
                if step >= max_steps:
                    break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "step": step,
            "act_freq_ema": act_freq_ema.detach().cpu(),
        },
        checkpoint_dir / "topk_sae_baseline.pt",
    )
    write_json(results_dir / "train_config_resolved.json", config)
    print(f"trained TopKSAE for {step} steps on {device}")
    print(f"wrote checkpoint to {checkpoint_dir / 'topk_sae_baseline.pt'}")
    print(f"wrote train log to {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
