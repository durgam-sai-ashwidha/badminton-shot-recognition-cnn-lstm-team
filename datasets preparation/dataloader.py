"""DataLoader Module for Badminton Shot Classification (Phase 1 / Phase 2).

Constructs and verifies PyTorch DataLoaders for train, validation, and test splits.
Provides deterministic validation/test pipelines, reproducible worker seeding,
and smoke-testing utilities for CNN-LSTM input validation.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

# Import BadmintonShotDataset from sibling module
try:
    from badminton_dataset import BadmintonShotDataset, CANONICAL_CLASSES
except ImportError:
    from .badminton_dataset import BadmintonShotDataset, CANONICAL_CLASSES

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dataloader")


def seed_worker(worker_id: int) -> None:
    """Ensures deterministic random state across PyTorch DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


def get_badminton_dataloaders(
    manifest_path: str | Path,
    batch_size: int = 16,
    num_workers: int = 0,
    num_frames: int = 16,
    frame_size: Tuple[int, int] = (224, 224),
    sampling_strategy: str = "uniform",
    transform_train: Optional[Callable] = None,
    transform_val: Optional[Callable] = None,
    pin_memory: bool = False,
    seed: int = 42,
    base_dir: Optional[str | Path] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Constructs DataLoaders for train, validation, and test splits.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    manifest = Path(manifest_path).resolve()
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    # Set torch generator for reproducible shuffling
    gen = torch.Generator()
    gen.manual_seed(seed)

    # Train dataset
    train_dataset = BadmintonShotDataset(
        manifest_path=manifest,
        split="train",
        num_frames=num_frames,
        frame_size=frame_size,
        sampling_strategy=sampling_strategy,
        transform=transform_train,
        base_dir=base_dir,
    )

    # Validation dataset
    val_dataset = BadmintonShotDataset(
        manifest_path=manifest,
        split="val",
        num_frames=num_frames,
        frame_size=frame_size,
        sampling_strategy=sampling_strategy,
        transform=transform_val,
        base_dir=base_dir,
    )

    # Test dataset
    test_dataset = BadmintonShotDataset(
        manifest_path=manifest,
        split="test",
        num_frames=num_frames,
        frame_size=frame_size,
        sampling_strategy=sampling_strategy,
        transform=transform_val,
        base_dir=base_dir,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=gen,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    logger.info(
        "DataLoaders constructed: train=%d samples (%d batches), val=%d samples (%d batches), test=%d samples (%d batches)",
        len(train_dataset), len(train_loader),
        len(val_dataset), len(val_loader),
        len(test_dataset), len(test_loader),
    )

    return train_loader, val_loader, test_loader


def smoke_test_dataloaders(
    manifest_path: str | Path,
    batch_size: int = 4,
    num_frames: int = 16,
    frame_size: Tuple[int, int] = (224, 224),
    max_test_batches: int = 2,
) -> Dict[str, Any]:
    """Runs a smoke test on the DataLoaders to verify batch tensor shapes and labels."""
    logger.info("Executing DataLoader smoke test with batch_size=%d, num_frames=%d...", batch_size, num_frames)

    train_loader, val_loader, test_loader = get_badminton_dataloaders(
        manifest_path=manifest_path,
        batch_size=batch_size,
        num_workers=0,
        num_frames=num_frames,
        frame_size=frame_size,
    )

    results: Dict[str, Any] = {
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "batches_verified": 0,
        "verified_shapes": [],
        "label_ranges_valid": True,
        "smoke_test_passed": False,
    }

    loaders = [("train", train_loader), ("val", val_loader), ("test", test_loader)]

    for name, loader in loaders:
        if len(loader.dataset) == 0:
            logger.warning("Split '%s' is empty, skipping smoke test batch verification.", name)
            continue

        batch_count = 0
        for batch in loader:
            batch_count += 1
            frames = batch["frames"]
            labels = batch["label"]

            # Expected shape: [B, T, C, H, W]
            expected_c = 3
            expected_h, expected_w = frame_size
            b, t, c, h, w = frames.shape

            if c != expected_c or h != expected_h or w != expected_w or t != num_frames:
                raise ValueError(
                    f"Unexpected tensor shape in {name} batch: {frames.shape}. Expected [B, {num_frames}, 3, {expected_h}, {expected_w}]"
                )

            # Check label bounds [0, 4]
            if (labels < 0).any() or (labels >= len(CANONICAL_CLASSES)).any():
                results["label_ranges_valid"] = False
                raise ValueError(f"Label out of bounds [0, {len(CANONICAL_CLASSES)-1}] in {name}: {labels.tolist()}")

            results["verified_shapes"].append(f"{name}_batch_{batch_count}: {list(frames.shape)}")
            results["batches_verified"] += 1

            if batch_count >= max_test_batches:
                break

    results["smoke_test_passed"] = results["batches_verified"] > 0 and results["label_ranges_valid"]
    logger.info("DataLoader smoke test PASSED: verified %d batches across splits.", results["batches_verified"])
    return results


def main() -> None:
    """CLI entry point for DataLoader verification."""
    parser = argparse.ArgumentParser(description="Build and Verify Badminton Shot DataLoaders.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="Phase -1 Data preparation/report/split_manifest.csv",
        help="Path to split_manifest.csv",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--test_smoke", action="store_true", help="Run quick smoke test")

    args = parser.parse_args()

    if args.test_smoke:
        results = smoke_test_dataloaders(
            manifest_path=args.manifest,
            batch_size=args.batch_size,
            num_frames=args.num_frames,
        )
        print("\n--- DATALOADER SMOKE TEST SUMMARY ---")
        print(f"Smoke test passed:       {results['smoke_test_passed']}")
        print(f"Train samples:           {results['train_samples']}")
        print(f"Validation samples:      {results['val_samples']}")
        print(f"Test samples:            {results['test_samples']}")
        print(f"Batches verified:        {results['batches_verified']}")
        print(f"Verified shapes:         {results['verified_shapes']}")
    else:
        train_l, val_l, test_l = get_badminton_dataloaders(
            manifest_path=args.manifest,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_frames=args.num_frames,
        )
        print("\n--- DATALOADERS CREATED ---")
        print(f"Train loader batches:    {len(train_l)} ({len(train_l.dataset)} samples)")
        print(f"Val loader batches:      {len(val_l)} ({len(val_l.dataset)} samples)")
        print(f"Test loader batches:     {len(test_l)} ({len(test_l.dataset)} samples)")


if __name__ == "__main__":
    main()
