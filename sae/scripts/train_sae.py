#!/usr/bin/env python3
"""Train the baseline TopK SAE."""

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

    train_x = torch.load(data_dir / "sae_train.pt", map_location="cpu")
    val_x = torch.load(data_dir / "sae_val.pt", map_location="cpu")
    train_loader = DataLoader(
        TensorDataset(train_x),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        drop_last=True,
    )

    model = TopKSAE(
        input_dim=int(config["input_dim"]),
        dict_size=int(config["dict_size"]),
        k=int(config["k"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))

    max_steps = int(config["training"]["max_steps"])
    log_every = int(config["training"].get("log_every", 100))
    eval_every = int(config["training"].get("eval_every", 500))
    checkpoint_every = int(config["training"].get("checkpoint_every", 1000))

    log_path = results_dir / "train_log.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "train_mse", "val_mse", "val_explained_variance"])
        writer.writeheader()

        step = 0
        while step < max_steps:
            for (batch,) in train_loader:
                step += 1
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                x_hat, _ = model(batch)
                loss = reconstruction_mse(batch, x_hat)
                loss.backward()
                optimizer.step()

                should_eval = step == 1 or step % eval_every == 0 or step == max_steps
                should_log = step == 1 or step % log_every == 0 or should_eval
                val_mse = ""
                val_ev = ""
                if should_eval:
                    with torch.no_grad():
                        val_batch = val_x[: min(len(val_x), int(config["training"]["batch_size"]) * 4)].to(device)
                        val_hat, _ = model(val_batch)
                        val_mse = float(reconstruction_mse(val_batch, val_hat).item())
                        val_ev = float(explained_variance(val_batch, val_hat).item())
                if should_log:
                    writer.writerow(
                        {
                            "step": step,
                            "train_mse": float(loss.item()),
                            "val_mse": val_mse,
                            "val_explained_variance": val_ev,
                        }
                    )
                    f.flush()
                if step % checkpoint_every == 0 or step == max_steps:
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "config": config,
                            "step": step,
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
