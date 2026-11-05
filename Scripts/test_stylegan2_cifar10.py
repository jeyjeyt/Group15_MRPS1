# copy of Group07's test_stylegan2.py, adapted to NVIDIA's
# class-conditional StyleGAN2-ADA CIFAR-10 checkpoint (cifar10.pkl).
# Differences from the CelebA script:
#   - metric defaults point to CIFAR-10 at 32x32
#   - samples are generated class-balanced (same number per class), then shuffled
#   - truncation_psi defaults to 1.0 (no truncation, the standard setting for FID)
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Models.pretrained_wrappers import StyleGAN2Wrapper
from Perturbation.pipeline_perturbations import add_perturbation_args, get_domain_shift_override
from Scripts.evaluation_runtime import file_signature, run_cached_evaluation
from Scripts.test_runtime_utils import (
    PreparedTestRun,
    add_common_test_args,
    close_prepared_test_run,
    get_generation_seed,
    initialize_test_run,
    resolve_reference_request,
)

NUM_CLASSES = 10


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate class-balanced samples from the pretrained StyleGAN2-ADA CIFAR-10 checkpoint."
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/StyleGAN/CIFAR10/stylegan2_generator.pkl")
    parser.add_argument("--out-dir", type=str, default="outputs/stylegan2_cifar10_test")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--truncation-psi", type=float, default=1.0)
    parser.add_argument("--noise-mode", type=str, default="const", choices=["const", "random", "none"])
    parser.add_argument(
        "--class-idx",
        type=int,
        default=None,
        help="Generate only this class. Default (None) generates all classes equally.",
    )
    parser.add_argument("--cuda", action="store_true")
    add_common_test_args(
        parser,
        metrics_dataset_default="cifar10",
        metrics_dataset_choices=["mnist", "cifar10", "celeba", "chestxray14"],
        metrics_data_root_default="data/CIFAR10",
        metrics_image_size_default=32,
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on errors (missing checkpoint/loading/sampling) instead of skipping.",
    )
    add_perturbation_args(parser)
    return parser.parse_args(argv)


def _resolve_real_reference_request(args: argparse.Namespace, default_image_size: int):
    return resolve_reference_request(args, default_image_size, get_domain_shift_override(args))


def _build_generation_payload(args: argparse.Namespace, checkpoint):
    return {
        "model_name": "stylegan2",
        "dataset": "cifar10",
        "checkpoint": file_signature(checkpoint),
        "num_samples": int(args.num_samples),
        "batch_size": int(args.batch_size),
        "truncation_psi": float(args.truncation_psi),
        "noise_mode": str(args.noise_mode),
        "class_idx": None if args.class_idx is None else int(args.class_idx),
        "class_balanced": args.class_idx is None,
        "generation_seed": get_generation_seed(args),
    }


def _class_plan(wrapper: StyleGAN2Wrapper, total_samples: int, class_idx: int | None):
    # one fixed class, or an equal share for every class (remainder goes to the first classes)
    if class_idx is not None:
        return [(int(class_idx), int(total_samples))]
    num_classes = int(getattr(wrapper.generator, "c_dim", 0) or 0) or NUM_CLASSES
    base, extra = divmod(int(total_samples), num_classes)
    return [(label, base + (1 if label < extra else 0)) for label in range(num_classes)]


def _generate_in_batches(
    *,
    wrapper: StyleGAN2Wrapper,
    total_samples: int,
    batch_size: int,
    device: torch.device,
    truncation_psi: float,
    noise_mode: str,
    class_idx: int | None,
    seed: int | None,
):
    samples: list[torch.Tensor] = []
    offset = 0

    for label, count in _class_plan(wrapper, total_samples, class_idx):
        remaining = int(count)
        while remaining > 0:
            this_batch = min(int(batch_size), remaining)
            batch_seed = None if seed is None else int(seed + offset)
            generated = wrapper.sample(
                this_batch,
                device=device,
                truncation_psi=truncation_psi,
                noise_mode=noise_mode,
                class_idx=label,
                seed=batch_seed,
            )
            samples.append(generated)
            remaining -= this_batch
            offset += this_batch

    images = torch.cat(samples, dim=0)
    # shuffle so classes are mixed (otherwise the preview image shows only class 0)
    shuffle_generator = torch.Generator().manual_seed(0 if seed is None else int(seed))
    order = torch.randperm(images.shape[0], generator=shuffle_generator)
    return images[order]


def prepare_run(args):
    initialize_test_run(args, context="test_stylegan2_cifar10")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        message = f"StyleGAN2 CIFAR-10 test skipped: checkpoint not found at {checkpoint}"
        if args.strict:
            raise FileNotFoundError(message)
        print(message)
        return None

    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    wrapper = StyleGAN2Wrapper(checkpoint_path=str(checkpoint))
    return PreparedTestRun(
        model_name="stylegan2",
        generation_payload=_build_generation_payload(args, checkpoint),
        generate_samples=lambda total_samples=None: _generate_in_batches(
            wrapper=wrapper,
            total_samples=int(args.num_samples) if total_samples is None else int(total_samples),
            batch_size=max(1, int(args.batch_size)),
            device=device,
            truncation_psi=float(args.truncation_psi),
            noise_mode=str(args.noise_mode),
            class_idx=args.class_idx,
            seed=get_generation_seed(args),
        ),
        resolve_reference_request=_resolve_real_reference_request,
        cleanup=None,
    )


def run_with_args(
    args: argparse.Namespace,
    *,
    prepared=None,
    session=None,
    write_preview=True,
    persist_derived_feature_artifacts=True,
    bootstrap_samples_override=None,
    bootstrap_policy="full",
):
    owned_prepared = prepared is None
    try:
        prepared_run = prepared or prepare_run(args)
        if prepared_run is None:
            return None
        return run_cached_evaluation(
            args=args,
            model_name=prepared_run.model_name,
            generation_payload=prepared_run.generation_payload,
            generate_samples=prepared_run.generate_samples,
            resolve_reference_request=prepared_run.resolve_reference_request,
            session=session,
            write_preview=write_preview,
            persist_derived_feature_artifacts=persist_derived_feature_artifacts,
            bootstrap_samples_override=bootstrap_samples_override,
            bootstrap_policy=bootstrap_policy,
        )
    except Exception as exc:
        print(f"StyleGAN2 CIFAR-10 test failed: {exc}")
        if args.strict:
            raise SystemExit(1)
        return None
    finally:
        if owned_prepared:
            close_prepared_test_run(prepared if prepared is not None else locals().get("prepared_run"))


def main():
    args = parse_args()
    run_with_args(args)


if __name__ == "__main__":
    main()