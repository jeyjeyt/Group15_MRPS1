# M1.1 — DDPM/CIFAR-10 baseline

This milestone reproduces and validates the previous group's DDPM/CIFAR-10
baseline using the previous team's existing pipeline and code. The implementation
uses the pretrained `google/ddpm-cifar10-32` checkpoint and the Inception-v3
feature implementation from the previous project.

The seven required metrics are:

- FID
- KID
- Inception Score
- Precision
- Recall
- Density
- Coverage

The real run used 1280 generated samples and 1280 real CIFAR-10 reference
samples, generation seed 1, reference seed 10, batch size 64, and
Inception-v3 features.

Run the same baseline from the previous project:

```bash
cd TestonPreviousGroupCode/_group07_embedded_project/Group07_Improving_Metrics-main
CUDA_VISIBLE_DEVICES=1 /home/i6470689/Group15_MRPS1/.venv/bin/python run_baseline.py \
  --num-samples 1280 \
  --metrics-samples 1280 \
  --batch-size 64 \
  --seed 1 \
  --cuda
```

The baseline report is:
`outputs/baseline_ddpm_cifar10_seed1/metrics_report.json`.

## Reproducible smoke validation

No model download is needed:

```bash
python3 milestones/M1_1_ddpm_cifar10_baseline/run_m1_1.py --smoke
```

This writes `results/smoke_metrics.json` and
`results/smoke_metrics.csv`. It validates all seven formulas and the output
schema, but is **not** a DDPM result.

## Real feature evaluation

Install the project's experiment dependencies, generate the DDPM samples and
extract matching real/fake Inception-v3 features. Then run:

```bash
python3 milestones/M1_1_ddpm_cifar10_baseline/run_m1_1.py \
  --real-features path/to/real_features.npz \
  --fake-features path/to/fake_features.npz \
  --output-dir milestones/M1_1_ddpm_cifar10_baseline/results/ddpm_cifar10
```

Each NPZ must contain a `features` array with shape `(n, d)`. For Inception
Score, the fake NPZ must additionally contain `logits` with shape `(n, k)`.
The runner fails explicitly when required inputs are absent; it never
substitutes synthetic values for a real experiment.
