#!/usr/bin/env python3
"""Validate Phase 1 PRISM embedding artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


REQUIRED_EXTRA_KEYS = {
    "chosen_conv_embedding",
    "rejected_conv_embedding",
    "user_id",
    "dialog_id",
    "seen",
    "split",
}


def _as_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.as_tensor(value).detach().cpu().float()


def validate_file(path: Path, expected_split: str, hidden_dim: int) -> bool:
    rows = torch.load(path, map_location="cpu")
    ok = True
    chosen_norms = []
    rejected_norms = []
    diff_norms = []
    identical_pairs = 0
    nonfinite_values = 0

    print(f"{expected_split}_path: {path}")
    print(f"{expected_split}_record_count: {len(rows)}")

    for idx, row in enumerate(rows):
        extra = row.get("extra_info")
        if not isinstance(extra, dict):
            print(f"ERROR {expected_split}[{idx}]: missing dict extra_info")
            return False

        missing = REQUIRED_EXTRA_KEYS.difference(extra)
        if missing:
            print(f"ERROR {expected_split}[{idx}]: missing keys {sorted(missing)}")
            ok = False
            continue

        if extra["split"] != expected_split:
            print(
                f"ERROR {expected_split}[{idx}]: split={extra['split']!r}, "
                f"expected {expected_split!r}"
            )
            ok = False

        chosen = _as_tensor(extra["chosen_conv_embedding"])
        rejected = _as_tensor(extra["rejected_conv_embedding"])

        for label, tensor in (("chosen", chosen), ("rejected", rejected)):
            if tensor.ndim != 1 or tensor.numel() != hidden_dim:
                print(
                    f"ERROR {expected_split}[{idx}]: {label} shape "
                    f"{tuple(tensor.shape)}, expected ({hidden_dim},)"
                )
                ok = False
            finite = torch.isfinite(tensor)
            if not bool(finite.all()):
                nonfinite_values += int((~finite).sum().item())
                ok = False

        if chosen.shape == rejected.shape and torch.allclose(chosen, rejected):
            identical_pairs += 1

        if chosen.ndim == 1 and chosen.numel() == hidden_dim:
            chosen_norms.append(torch.linalg.vector_norm(chosen))
        if rejected.ndim == 1 and rejected.numel() == hidden_dim:
            rejected_norms.append(torch.linalg.vector_norm(rejected))
        if chosen.shape == rejected.shape:
            diff_norms.append(torch.linalg.vector_norm(chosen - rejected))

    total_pairs = max(1, len(rows))
    identical_rate = identical_pairs / total_pairs
    if identical_rate > 0.01:
        print(
            f"ERROR {expected_split}: identical chosen/rejected pairs "
            f"{identical_pairs}/{len(rows)} ({identical_rate:.4%})"
        )
        ok = False

    def print_stats(name: str, values: list[torch.Tensor]) -> None:
        if not values:
            print(f"{expected_split}_{name}: no valid values")
            return
        tensor = torch.stack(values)
        print(f"{expected_split}_{name}_mean: {tensor.mean().item():.6f}")
        print(f"{expected_split}_{name}_std: {tensor.std(unbiased=False).item():.6f}")
        print(f"{expected_split}_{name}_min: {tensor.min().item():.6f}")
        print(f"{expected_split}_{name}_max: {tensor.max().item():.6f}")

    print_stats("chosen_norm", chosen_norms)
    print_stats("rejected_norm", rejected_norms)
    print_stats("chosen_minus_rejected_norm", diff_norms)
    print(f"{expected_split}_identical_pair_count: {identical_pairs}")
    print(f"{expected_split}_nonfinite_value_count: {nonfinite_values}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/prism")
    parser.add_argument("--hidden-dim", type=int, default=4096)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    checks = (
        validate_file(data_dir / "train_embeddings.pkl", "train", args.hidden_dim),
        validate_file(data_dir / "test_embeddings.pkl", "test", args.hidden_dim),
    )

    if all(checks):
        print("embedding_validation: PASS")
        return 0
    print("embedding_validation: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
