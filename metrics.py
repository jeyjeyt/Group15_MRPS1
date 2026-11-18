"""NumPy implementations of the seven M1.1 evaluation metrics."""

from __future__ import annotations

import numpy as np


def _validate_pair(real: np.ndarray, fake: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    real = np.asarray(real, dtype=np.float64)
    fake = np.asarray(fake, dtype=np.float64)
    if real.ndim != 2 or fake.ndim != 2 or real.shape[1] != fake.shape[1]:
        raise ValueError("real and fake features must be 2-D arrays with equal width")
    if len(real) < 2 or len(fake) < 2:
        raise ValueError("at least two real and fake samples are required")
    return real, fake


def _cov(features: np.ndarray) -> np.ndarray:
    return np.atleast_2d(np.cov(features, rowvar=False))


def fid(real: np.ndarray, fake: np.ndarray) -> float:
    real, fake = _validate_pair(real, fake)
    mean_delta = real.mean(0) - fake.mean(0)
    cov_real, cov_fake = _cov(real), _cov(fake)
    real_eigenvalues, real_eigenvectors = np.linalg.eigh(cov_real)
    real_root = real_eigenvectors @ np.diag(
        np.sqrt(np.clip(real_eigenvalues, 0, None))
    ) @ real_eigenvectors.T
    middle = real_root @ cov_fake @ real_root
    middle_eigenvalues = np.linalg.eigvalsh((middle + middle.T) / 2)
    cross_covariance_trace = np.sqrt(np.clip(middle_eigenvalues, 0, None)).sum()
    value = mean_delta @ mean_delta + np.trace(cov_real) + np.trace(cov_fake)
    return float(max(0.0, value - 2 * cross_covariance_trace))


def kid(real: np.ndarray, fake: np.ndarray) -> float:
    real, fake = _validate_pair(real, fake)
    dimension = real.shape[1]
    kernel_rr = (real @ real.T / dimension + 1) ** 3
    kernel_ff = (fake @ fake.T / dimension + 1) ** 3
    kernel_rf = (real @ fake.T / dimension + 1) ** 3
    rr = (kernel_rr.sum() - np.trace(kernel_rr)) / (len(real) * (len(real) - 1))
    ff = (kernel_ff.sum() - np.trace(kernel_ff)) / (len(fake) * (len(fake) - 1))
    return float(rr + ff - 2 * kernel_rf.mean())


def inception_score(logits: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or len(logits) < 2:
        raise ValueError("logits must be a 2-D array with at least two samples")
    logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    marginal = probabilities.mean(axis=0)
    kl = probabilities * (np.log(probabilities + 1e-12) - np.log(marginal + 1e-12))
    return float(np.exp(kl.sum(axis=1).mean()))


def _radii(features: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=2)
    distances[np.diag_indices_from(distances)] = np.inf
    return np.partition(distances, 2, axis=1)[:, 1]


def precision_recall_density_coverage(
    real: np.ndarray, fake: np.ndarray
) -> tuple[float, float, float, float]:
    real, fake = _validate_pair(real, fake)
    real_radii, fake_radii = _radii(real), _radii(fake)
    fake_to_real = np.linalg.norm(fake[:, None, :] - real[None, :, :], axis=2)
    real_to_fake = fake_to_real.T
    precision_hits = (fake_to_real <= real_radii[None, :]).any(axis=1)
    recall_hits = (real_to_fake <= fake_radii[None, :]).any(axis=1)
    density = float((fake_to_real <= real_radii[None, :]).sum() / (len(fake) * len(real)))
    coverage = float((fake_to_real.min(axis=0) <= real_radii).mean())
    return (
        float(precision_hits.mean()),
        float(recall_hits.mean()),
        density,
        coverage,
    )


def all_metrics(
    real: np.ndarray, fake: np.ndarray, logits: np.ndarray
) -> dict[str, float]:
    precision, recall, density, coverage = precision_recall_density_coverage(real, fake)
    return {
        "fid": fid(real, fake),
        "kid": kid(real, fake),
        "inception_score": inception_score(logits),
        "precision": precision,
        "recall": recall,
        "density": density,
        "coverage": coverage,
    }
