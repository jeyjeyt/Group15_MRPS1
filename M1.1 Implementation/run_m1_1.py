"""Run the M1.1 seven-metric baseline evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import all_metrics


METRICS = ("fid", "kid", "inception_score", "precision", "recall", "density", "coverage")


def _smoke_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2026)
    real = rng.normal(0, 1, size=(128, 16))
    fake = rng.normal(0.15, 1.05, size=(128, 16))
    logits = rng.normal(0, 1, size=(128, 10))
    return real, fake, logits


def _load(path: Path, require_logits: bool) -> tuple[np.ndarray, np.ndarray | None]:
    if not path.exists():
        raise FileNotFoundError(f"feature file does not exist: {path}")
    data = np.load(path)
    if "features" not in data:
        raise ValueError(f"{path} must contain an array named 'features'")
    logits = data["logits"] if require_logits and "logits" in data else None
    if require_logits and logits is None:
        raise ValueError(f"{path} must contain 'logits' for Inception Score")
    return np.asarray(data["features"]), logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run deterministic formula validation")
    parser.add_argument("--real-features", type=Path)
    parser.add_argument("--fake-features", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()

    if args.smoke:
        real, fake, logits = _smoke_inputs()
        name = "smoke"
    else:
        if args.real_features is None or args.fake_features is None:
            parser.error("--real-features and --fake-features are required unless --smoke is used")
        real, _ = _load(args.real_features, require_logits=False)
        fake, logits = _load(args.fake_features, require_logits=True)
        name = "ddpm_cifar10"
    results = all_metrics(real, fake, logits)
    if tuple(results) != METRICS:
        raise RuntimeError(f"unexpected metric schema: {tuple(results)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{name}_metrics.json"
    csv_path = args.output_dir / f"{name}_metrics.csv"
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("metric", "value"))
        writer.writeheader()
        writer.writerows({"metric": metric, "value": value} for metric, value in results.items())
    print(json.dumps(results, indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
