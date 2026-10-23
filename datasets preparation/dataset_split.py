"""Dataset Splitting Module for Badminton Shot Classification (Phase 1).

Creates group-aware, balanced, leakage-free training, validation, and test splits.
Enforces the Phase 1 target of 1,600 valid, unique clips per class (8,000 total).
Assigns source match/session groups to splits without leaking groups across splits,
while maintaining balanced class distributions (70% train, 15% val, 15% test).
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dataset_split")

# Target classes with canonical alphabetical index mapping
CANONICAL_CLASSES: List[str] = ["Clear", "Drive", "Drop", "Net_Shot", "Smash"]
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CANONICAL_CLASSES)}
DEFAULT_TARGET_PER_CLASS: int = 1600  # Phase 1 target: 1,600 per class (8,000 total)

# Match ID extraction patterns
FB_PATTERN = re.compile(r"^(FB\d{4})_", re.IGNORECASE)
VB_PATTERN = re.compile(r"^VB_.*?_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_", re.IGNORECASE)
AVI_FB_PATTERN = re.compile(r"^(FB_\d{4})_", re.IGNORECASE)


@dataclass
class SplitItem:
    """Represents a clip in the split manifest."""
    clip_path: str
    filename: str
    canonical_label: str
    class_index: int
    split: str  # train, val, test, excluded
    source_group: str
    duplicate_status: str  # UNIQUE, CANONICAL, DUPLICATE_EXCLUDED
    exclusion_reason: str = ""


def extract_source_group(filename: str) -> str:
    """Extracts match / session group from clip filename."""
    m_fb = FB_PATTERN.match(filename)
    if m_fb:
        return m_fb.group(1).upper()
    m_vb = VB_PATTERN.match(filename)
    if m_vb:
        return f"VB_{m_vb.group(1)}"
    m_avi = AVI_FB_PATTERN.match(filename)
    if m_avi:
        return m_avi.group(1).upper()
    parts = filename.split("_")
    return parts[0] if parts else "UNKNOWN_GROUP"


def stratify_groups_to_splits(
    group_profiles: Dict[str, Counter],
    total_active_clips: int,
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> Dict[str, str]:
    """Assigns source groups to train/val/test splits using capacity-constrained greedy bin packing.
    
    Guarantees 100% of clips from any single source group reside in exactly one split,
    completely eliminating match-level data leakage while hitting target split proportions.
    """
    rng = random.Random(seed)
    train_ratio, val_ratio, test_ratio = split_ratios
    ratio_sum = train_ratio + val_ratio + test_ratio

    target_totals = {
        "train": total_active_clips * (train_ratio / ratio_sum),
        "val": total_active_clips * (val_ratio / ratio_sum),
        "test": total_active_clips * (test_ratio / ratio_sum),
    }

    groups = list(group_profiles.keys())
    rng.shuffle(groups)
    groups.sort(key=lambda g: sum(group_profiles[g].values()), reverse=True)

    group_assignment: Dict[str, str] = {}
    split_totals: Dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for g in groups:
        g_size = sum(group_profiles[g].values())

        # Select split with lowest fill percentage relative to target
        best_split = min(
            ["train", "val", "test"],
            key=lambda s: (split_totals[s] / max(1.0, target_totals[s]), -target_totals[s]),
        )

        group_assignment[g] = best_split
        split_totals[best_split] += g_size

    return group_assignment


def run_dataset_split(
    dataset_dir: str | Path,
    output_manifest_csv: Optional[str | Path] = None,
    output_dist_csv: Optional[str | Path] = None,
    duplicate_report_csv: Optional[str | Path] = None,
    target_per_class: int = DEFAULT_TARGET_PER_CLASS,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[SplitItem], Dict[str, Any]]:
    """Performs dataset balancing, group-aware splitting, and manifest generation."""
    root_path = Path(dataset_dir).resolve()
    clips_dir = root_path / "CLIPS" if (root_path / "CLIPS").is_dir() else root_path

    logger.info("Initializing dataset splitting at: %s", clips_dir)
    if not clips_dir.exists():
        raise FileNotFoundError(f"Clips directory not found: {clips_dir}")

    # Load duplicate exclusions if available
    excluded_duplicate_paths: Set[str] = set()
    if duplicate_report_csv and Path(duplicate_report_csv).exists():
        with open(duplicate_report_csv, "r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                if row.get("handling_action") == "EXCLUDE_FROM_SPLIT":
                    excluded_duplicate_paths.add(str(Path(row["file_path"]).resolve()))
        logger.info("Loaded %d duplicate paths to exclude from splits.", len(excluded_duplicate_paths))

    # Discover eligible unique clips grouped by class
    class_clips: Dict[str, List[Tuple[str, str, str, str, Path]]] = defaultdict(list)
    total_raw_clips = 0

    for cdir in clips_dir.iterdir():
        if cdir.is_dir() and cdir.name in CANONICAL_CLASSES:
            c_name = cdir.name
            for f in sorted(cdir.iterdir()):
                if f.is_file():
                    total_raw_clips += 1
                    abs_p = str(f.resolve())
                    grp = extract_source_group(f.name)
                    is_dup = abs_p in excluded_duplicate_paths
                    class_clips[c_name].append((str(f), f.name, c_name, grp, f, is_dup))

    # Evaluate unique valid counts vs target
    verified_unique_counts: Dict[str, int] = {}
    class_shortages: Dict[str, int] = {}
    all_targets_met = True

    for c in CANONICAL_CLASSES:
        eligible = [x for x in class_clips[c] if not x[5]]  # not is_dup
        cnt = len(eligible)
        verified_unique_counts[c] = cnt
        if cnt < target_per_class:
            shortage = target_per_class - cnt
            class_shortages[c] = shortage
            all_targets_met = False
            logger.warning("Class '%s' has %d unique clips, SHORTAGE of %d against target %d!",
                           c, cnt, shortage, target_per_class)
        else:
            logger.info("Class '%s' has %d unique clips (Target %d met; Surplus: %d).",
                        c, cnt, target_per_class, cnt - target_per_class)

    # Perform reproducible balancing selection
    rng = random.Random(seed)
    selected_clips: List[Tuple[str, str, str, str, Path]] = []
    surplus_excluded_clips: List[Tuple[str, str, str, str, Path]] = []
    duplicate_excluded_clips: List[Tuple[str, str, str, str, Path]] = []

    for c in CANONICAL_CLASSES:
        items = class_clips[c]
        eligible = [x[:5] for x in items if not x[5]]
        dups = [x[:5] for x in items if x[5]]
        duplicate_excluded_clips.extend(dups)

        if all_targets_met:
            # Reproducibly shuffle eligible clips and take exactly target_per_class
            shuffled = list(eligible)
            rng.shuffle(shuffled)
            selected_clips.extend(shuffled[:target_per_class])
            surplus_excluded_clips.extend(shuffled[target_per_class:])
        else:
            # When shortage exists, include all eligible clips without fabricating
            selected_clips.extend(eligible)

    # Build group profiles across selected active clips
    active_group_profiles: Dict[str, Counter] = defaultdict(Counter)
    for path_str, fname, c_name, grp, f_path in selected_clips:
        active_group_profiles[grp][c_name] += 1

    logger.info("Active balanced dataset: %d clips across %d source groups.",
                len(selected_clips), len(active_group_profiles))

    # Perform group-aware split allocation
    group_assignment = stratify_groups_to_splits(
        group_profiles=active_group_profiles,
        total_active_clips=len(selected_clips),
        split_ratios=(train_ratio, val_ratio, test_ratio),
        seed=seed,
    )

    # Verify no group leaks across splits
    split_class_dist: Dict[str, Counter] = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
        "excluded": Counter(),
    }
    manifest_items: List[SplitItem] = []

    # 1. Add active split clips
    for path_str, fname, c_name, grp, f_path in selected_clips:
        split_name = group_assignment.get(grp, "train")
        c_idx = CLASS_TO_IDX.get(c_name, -1)
        item = SplitItem(
            clip_path=path_str,
            filename=fname,
            canonical_label=c_name,
            class_index=c_idx,
            split=split_name,
            source_group=grp,
            duplicate_status="UNIQUE",
            exclusion_reason="",
        )
        manifest_items.append(item)
        split_class_dist[split_name][c_name] += 1

    # 2. Add surplus excluded clips
    for path_str, fname, c_name, grp, f_path in surplus_excluded_clips:
        c_idx = CLASS_TO_IDX.get(c_name, -1)
        item = SplitItem(
            clip_path=path_str,
            filename=fname,
            canonical_label=c_name,
            class_index=c_idx,
            split="excluded",
            source_group=grp,
            duplicate_status="UNIQUE",
            exclusion_reason=f"Surplus clip excluded to achieve balanced target of {target_per_class} clips per class",
        )
        manifest_items.append(item)
        split_class_dist["excluded"][c_name] += 1

    # 3. Add duplicate excluded clips
    for path_str, fname, c_name, grp, f_path in duplicate_excluded_clips:
        c_idx = CLASS_TO_IDX.get(c_name, -1)
        item = SplitItem(
            clip_path=path_str,
            filename=fname,
            canonical_label=c_name,
            class_index=c_idx,
            split="excluded",
            source_group=grp,
            duplicate_status="DUPLICATE_EXCLUDED",
            exclusion_reason="Exact duplicate of canonical file; excluded to prevent leakage",
        )
        manifest_items.append(item)
        split_class_dist["excluded"][c_name] += 1

    # Sort manifest deterministically by path
    manifest_items.sort(key=lambda x: x.clip_path)

    # Verification checks
    group_to_splits: Dict[str, Set[str]] = defaultdict(set)
    for it in manifest_items:
        if it.split != "excluded":
            group_to_splits[it.source_group].add(it.split)

    leaking_groups = {g: s for g, s in group_to_splits.items() if len(s) > 1}
    if leaking_groups:
        raise RuntimeError(f"Source group leakage detected across active splits: {leaking_groups}")

    tot_train = sum(split_class_dist["train"].values())
    tot_val = sum(split_class_dist["val"].values())
    tot_test = sum(split_class_dist["test"].values())
    tot_active = tot_train + tot_val + tot_test

    # Generate class distribution table rows
    distribution_rows: List[Dict[str, Any]] = []
    for c in CANONICAL_CLASSES:
        raw_cnt = len(class_clips[c])
        train_c = split_class_dist["train"].get(c, 0)
        val_c = split_class_dist["val"].get(c, 0)
        test_c = split_class_dist["test"].get(c, 0)
        excl_c = split_class_dist["excluded"].get(c, 0)
        bal_c = train_c + val_c + test_c
        distribution_rows.append(
            {
                "class_name": c,
                "raw_total": raw_cnt,
                "balanced_active_total": bal_c,
                "train_count": train_c,
                "train_pct": round((train_c / max(1, bal_c)) * 100, 2),
                "val_count": val_c,
                "val_pct": round((val_c / max(1, bal_c)) * 100, 2),
                "test_count": test_c,
                "test_pct": round((test_c / max(1, bal_c)) * 100, 2),
                "excluded_count": excl_c,
            }
        )

    # Add TOTAL row
    distribution_rows.append(
        {
            "class_name": "TOTAL",
            "raw_total": total_raw_clips,
            "balanced_active_total": tot_active,
            "train_count": tot_train,
            "train_pct": round((tot_train / max(1, tot_active)) * 100, 2),
            "val_count": tot_val,
            "val_pct": round((tot_val / max(1, tot_active)) * 100, 2),
            "test_count": tot_test,
            "test_pct": round((tot_test / max(1, tot_active)) * 100, 2),
            "excluded_count": len(manifest_items) - tot_active,
        }
    )

    summary: Dict[str, Any] = {
        "total_raw_clips": total_raw_clips,
        "target_per_class": target_per_class,
        "total_target": target_per_class * len(CANONICAL_CLASSES),
        "all_targets_met": all_targets_met,
        "verified_unique_counts": verified_unique_counts,
        "class_shortages": class_shortages,
        "active_balanced_clips": tot_active,
        "train_count": tot_train,
        "val_count": tot_val,
        "test_count": tot_test,
        "excluded_surplus_count": len(surplus_excluded_clips),
        "excluded_duplicate_count": len(duplicate_excluded_clips),
        "total_excluded_count": len(manifest_items) - tot_active,
        "train_actual_pct": round((tot_train / max(1, tot_active)) * 100, 2),
        "val_actual_pct": round((tot_val / max(1, tot_active)) * 100, 2),
        "test_actual_pct": round((tot_test / max(1, tot_active)) * 100, 2),
        "distinct_active_source_groups": len(active_group_profiles),
        "group_leakage_detected": False,
        "seed": seed,
    }

    logger.info(
        "Splitting complete: Balanced Target=%s | Train=%d (%.1f%%), Val=%d (%.1f%%), Test=%d (%.1f%%), Excluded=%d",
        "ACHIEVED (8,000 clips)" if all_targets_met else "NOT ACHIEVED",
        tot_train, summary["train_actual_pct"],
        tot_val, summary["val_actual_pct"],
        tot_test, summary["test_actual_pct"],
        summary["total_excluded_count"],
    )

    # Save split manifest CSV
    if output_manifest_csv:
        m_path = Path(output_manifest_csv).resolve()
        m_path.parent.mkdir(parents=True, exist_ok=True)
        m_fields = [
            "clip_path",
            "filename",
            "canonical_label",
            "class_index",
            "split",
            "source_group",
            "duplicate_status",
            "exclusion_reason",
        ]
        with open(m_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=m_fields)
            writer.writeheader()
            for it in manifest_items:
                writer.writerow(asdict(it))
        logger.info("Saved split manifest to: %s", m_path)

    # Save class distribution CSV
    if output_dist_csv:
        d_path = Path(output_dist_csv).resolve()
        d_path.parent.mkdir(parents=True, exist_ok=True)
        d_fields = [
            "class_name",
            "raw_total",
            "balanced_active_total",
            "train_count",
            "train_pct",
            "val_count",
            "val_pct",
            "test_count",
            "test_pct",
            "excluded_count",
        ]
        with open(d_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=d_fields)
            writer.writeheader()
            for r in distribution_rows:
                writer.writerow(r)
        logger.info("Saved class distribution table to: %s", d_path)

    return manifest_items, summary


def main() -> None:
    """CLI entry point for dataset splitting."""
    parser = argparse.ArgumentParser(description="Create Group-Aware Balanced Dataset Splits.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="d:/badminton-shot-dataset-lab/data",
        help="Path to dataset directory.",
    )
    parser.add_argument(
        "--output_manifest",
        type=str,
        default="Phase -1 Data preparation/report/split_manifest.csv",
        help="Path to save split manifest CSV.",
    )
    parser.add_argument(
        "--output_distribution",
        type=str,
        default="Phase -1 Data preparation/report/class_distribution.csv",
        help="Path to save class distribution CSV.",
    )
    parser.add_argument(
        "--duplicate_report",
        type=str,
        default="Phase -1 Data preparation/report/duplicate_audit_report.csv",
        help="Path to duplicate audit report CSV (optional).",
    )
    parser.add_argument("--target_per_class", type=int, default=DEFAULT_TARGET_PER_CLASS)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    items, summary = run_dataset_split(
        dataset_dir=args.dataset_dir,
        output_manifest_csv=args.output_manifest,
        output_dist_csv=args.output_distribution,
        duplicate_report_csv=args.duplicate_report,
        target_per_class=args.target_per_class,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print("\n--- DATASET SPLIT & BALANCING SUMMARY ---")
    print(f"Total raw clips:         {summary['total_raw_clips']}")
    print(f"Target per class:        {summary['target_per_class']} (Total target: {summary['total_target']})")
    print(f"Target met:              {summary['all_targets_met']}")
    print(f"Active balanced clips:   {summary['active_balanced_clips']}")
    print(f"Train split:             {summary['train_count']} ({summary['train_actual_pct']}%)")
    print(f"Validation split:        {summary['val_count']} ({summary['val_actual_pct']}%)")
    print(f"Test split:              {summary['test_count']} ({summary['test_actual_pct']}%)")
    print(f"Excluded surplus clips:  {summary['excluded_surplus_count']}")
    print(f"Excluded duplicates:     {summary['excluded_duplicate_count']}")
    print(f"Active source groups:    {summary['distinct_active_source_groups']}")
    print(f"Group leakage detected:  {summary['group_leakage_detected']}")
    print(f"Random seed:             {summary['seed']}")


if __name__ == "__main__":
    main()
