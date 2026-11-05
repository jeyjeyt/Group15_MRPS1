# M1.2 – DCGAN and StyleGAN2 on CIFAR-10

Owner: Jessica

## Part 1: DCGAN on CIFAR-10 

**Setup**
- Model: DCGAN, pretrained weights from Group07 (`Models/saved_weights/netG_best.pth`)
- Dataset: CIFAR-10 (real reference)
- Samples: 1280 generated, 1280 real
- Generation seed: 1, reference seed: 10
- Feature extractor: Inception-v3
- Confidence intervals: 100 subsamples of 80%, 95% percentile interval
- No perturbations (clean baseline)
- Raw output: `dcgan_cifar10_seed1_metrics.json`

**Results (compared with Group07's report, Section 5.1.2)**

| Metric                | Group07| Group15   | Our 95% interval |
|-----------------------|--------|--------|------------------|
| FID (lower is better) | 53.212 | 53.017 | 56.68 – 58.38 |
| KID (lower is better) | 0.020 | 0.0209 | 0.0201 – 0.0217 |
| IS (higher is better) | 4.758 | 4.852 | 4.62 – 5.04 |
| Precision (higher is better) | 0.738 | 0.723 | 0.709 – 0.751 |
| Recall (higher is better) | 0.504 | 0.507 | 0.489 – 0.534 |
| Density (higher is better) | 0.648 | 0.670 | 0.632 – 0.723 |
| Coverage (higher is better) | 0.499 | 0.502 | 0.484 – 0.535 |

**Conclusion:** our values are very close to Group07's. For IS, Precision, Recall, Density and Coverage, Group07's value falls inside our 95% interval, so the small differences are normal random variation. For KID, Group07 reports only three decimals (0.020), so we cannot tell whether it falls inside our interval (0.0201 – 0.0217), but the values are nearly identical. For FID, both Group07's value and our own full-sample value fall outside the interval (see the note below). Overall, the pipeline reproduces Group07's DCGAN results on the cluster.

**Note on FID:** the full-sample FID (53.02) is outside its own interval (56.68 – 58.38). This is expected: the interval is computed on 80% subsets (fewer images), and FID is biased upward with fewer samples (Chong & Forsyth, 2020). Group07 observed the same (their Section 5.2). This is relevant for M2 (uncertainty intervals) and for the bias-corrected FID in A2.

**Technical note:** FID initially failed with the newest SciPy (`sqrtm() got an unexpected keyword argument 'disp'`). Fixed by installing `scipy<1.15` (now pinned in `requirements.txt`).

## Part 2: StyleGAN2 on CIFAR-10

**Setup**
- Model: NVIDIA StyleGAN2-ADA, class-conditional CIFAR-10 checkpoint (`cifar10.pkl`), staged in `checkpoints/StyleGAN/CIFAR10/`
- New script: `Scripts/test_stylegan2_cifar10.py` (copy of Group07's CelebA script, adapted)
  - class-balanced generation: 128 images per class (1280 total), shuffled
  - truncation_psi = 1.0 (no truncation, standard for FID)
  - metrics against CIFAR-10 at 32x32
- New pipeline step: `test_stylegan2_cifar10`
- Same evaluation settings as DCGAN (1280 samples, generation seed 1, reference seed 10, Inception-v3, 100 subsamples of 80%)
- Raw output: `stylegan2_cifar10_seed1_metrics.json`

**Results**

| Metric                 | DCGAN (ours)| StyleGAN2 (ours)| StyleGAN2 95% interval | DDPM (Group07, reference) |
|-------------------------------------------------------------------------|
| FID (lower is better) | 53.02 | 20.42 | 24.88 – 25.55 | 28.34 |
| KID (lower is better) | 0.0209 | 0.00023 | 0.00007 – 0.00048 | 0.006 |
| IS (higher is better) | 4.85 | 8.41 | 7.91 – 8.67 | 7.17 |
| Precision (higher is better) | 0.723 | 0.805 | 0.791 – 0.818 | 0.800 |
| Recall (higher is better) | 0.507 | 0.756 | 0.747 – 0.783 | 0.719 |
| Density (higher is better) | 0.670 | 1.019 | 0.968 – 1.076 | 0.971 |
| Coverage (higher is better) | 0.502 | 0.971 | 0.962 – 0.982 | 0.877 |

**Conclusion:** StyleGAN2 is the best of the three models on all seven metrics, for both fidelity (Precision, Density) and diversity (Recall, Coverage). The ranking StyleGAN2 > DDPM > DCGAN is consistent with the literature.

**Notes**
- As for DCGAN, the full-sample FID (20.42) lies outside its subsampling interval, due to the small-sample bias of FID (relevant for M2).
- NVIDIA reports FID 2.42 for this checkpoint using 50,000 samples; our 20.42 uses 1,280 samples. This large gap illustrates how strongly FID depends on sample size.
- KID is close to zero (0.00023), so its interval is wider than the value itself. This matters for any score that normalises by the baseline value (e.g. RWFAS reliability), relevant for M4.
- NVIDIA's custom CUDA kernels failed to compile on the cluster; the code falls back to the reference implementation (slower, same results).
