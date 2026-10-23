"""Dataset Inspection Module for Badminton Shot Classification (Phase 1).

Inspects video dataset directories, verifies directory structure, discovers
supported video clips, checks readability and extracts video metadata
(resolution, fps, frame count, duration), and generates a dataset inventory.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Attempt cv2 import
try:
    import cv2  # type: ignore
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dataset_inspect")

TARGET_CLASSES: List[str] = ["Clear", "Drive", "Drop", "Net_Shot", "Smash"]
SUPPORTED_EXTENSIONS: Set[str] = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Match ID extraction patterns
FB_PATTERN = re.compile(r"^(FB\d{4})_", re.IGNORECASE)
VB_PATTERN = re.compile(r"^VB_.*?_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_", re.IGNORECASE)
AVI_FB_PATTERN = re.compile(r"^(FB_\d{4})_", re.IGNORECASE)


@dataclass
class ClipMetadata:
    """Holds structural and video metadata for an individual clip."""
    file_path: str
    filename: str
    class_name: str
    file_format: str
    file_size_bytes: int
    duration_seconds: float
    frame_rate: float
    resolution: str
    frame_count: int
    readable: bool
    source_group: str
    error_message: str = ""


def extract_source_group(filename: str) -> str:
    """Extracts a source group / match / session identifier from a clip filename."""
    m_fb = FB_PATTERN.match(filename)
    if m_fb:
        return m_fb.group(1).upper()
    m_vb = VB_PATTERN.match(filename)
    if m_vb:
        return f"VB_{m_vb.group(1)}"
    m_avi = AVI_FB_PATTERN.match(filename)
    if m_avi:
        return m_avi.group(1).upper()
    # Fallback to prefix before first underscore or file stem
    parts = filename.split("_")
    if len(parts) > 1:
        return parts[0]
    return Path(filename).stem


def inspect_single_clip(
    file_path: Path,
    class_name: str,
    extract_video_info: bool = True,
) -> ClipMetadata:
    """Inspects a single clip file and extracts its properties."""
    filename = file_path.name
    ext = file_path.suffix.lower()
    source_group = extract_source_group(filename)

    if not file_path.exists():
        return ClipMetadata(
            file_path=str(file_path),
            filename=filename,
            class_name=class_name,
            file_format=ext,
            file_size_bytes=-1,
            duration_seconds=0.0,
            frame_rate=0.0,
            resolution="0x0",
            frame_count=0,
            readable=False,
            source_group=source_group,
            error_message="File does not exist",
        )

    try:
        size_bytes = file_path.stat().st_size
    except OSError as err:
        return ClipMetadata(
            file_path=str(file_path),
            filename=filename,
            class_name=class_name,
            file_format=ext,
            file_size_bytes=-1,
            duration_seconds=0.0,
            frame_rate=0.0,
            resolution="0x0",
            frame_count=0,
            readable=False,
            source_group=source_group,
            error_message=f"stat failed: {err}",
        )

    if size_bytes == 0:
        return ClipMetadata(
            file_path=str(file_path),
            filename=filename,
            class_name=class_name,
            file_format=ext,
            file_size_bytes=0,
            duration_seconds=0.0,
            frame_rate=0.0,
            resolution="0x0",
            frame_count=0,
            readable=False,
            source_group=source_group,
            error_message="Empty file (0 bytes)",
        )

    if ext not in SUPPORTED_EXTENSIONS:
        return ClipMetadata(
            file_path=str(file_path),
            filename=filename,
            class_name=class_name,
            file_format=ext,
            file_size_bytes=size_bytes,
            duration_seconds=0.0,
            frame_rate=0.0,
            resolution="0x0",
            frame_count=0,
            readable=False,
            source_group=source_group,
            error_message=f"Unsupported file format: {ext}",
        )

    if not extract_video_info or not HAS_CV2:
        return ClipMetadata(
            file_path=str(file_path),
            filename=filename,
            class_name=class_name,
            file_format=ext,
            file_size_bytes=size_bytes,
            duration_seconds=0.0,
            frame_rate=0.0,
            resolution="unknown",
            frame_count=0,
            readable=True,
            source_group=source_group,
            error_message="",
        )

    # OpenCV metadata extraction
    try:
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            return ClipMetadata(
                file_path=str(file_path),
                filename=filename,
                class_name=class_name,
                file_format=ext,
                file_size_bytes=size_bytes,
                duration_seconds=0.0,
                frame_rate=0.0,
                resolution="0x0",
                frame_count=0,
                readable=False,
                source_group=source_group,
                error_message="OpenCV failed to open video stream",
            )

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = (frame_count / fps) if (fps > 0 and frame_count > 0) else 0.0

        cap.release()

        return ClipMetadata(
            file_path=str(file_path),
            filename=filename,
            class_name=class_name,
            file_format=ext,
            file_size_bytes=size_bytes,
            duration_seconds=round(duration, 3),
            frame_rate=round(fps, 2),
            resolution=f"{width}x{height}",
            frame_count=frame_count,
            readable=True,
            source_group=source_group,
            error_message="",
        )
    except Exception as err:
        return ClipMetadata(
            file_path=str(file_path),
            filename=filename,
            class_name=class_name,
            file_format=ext,
            file_size_bytes=size_bytes,
            duration_seconds=0.0,
            frame_rate=0.0,
            resolution="0x0",
            frame_count=0,
            readable=False,
            source_group=source_group,
            error_message=f"Video decoding exception: {err}",
        )


def resolve_clips_dir(dataset_path: Path) -> Path:
    """Resolves whether dataset_path points to data/, CLIPS/, or root."""
    if (dataset_path / "CLIPS").is_dir():
        return dataset_path / "CLIPS"
    if (dataset_path / "data" / "CLIPS").is_dir():
        return dataset_path / "data" / "CLIPS"
    return dataset_path


def run_dataset_inspection(
    dataset_dir: str | Path,
    output_csv: Optional[str | Path] = None,
    extract_video_info: bool = True,
    max_workers: int = 8,
) -> Tuple[List[ClipMetadata], Dict[str, Any]]:
    """Runs a complete inspection over the dataset directory."""
    root_path = Path(dataset_dir).resolve()
    clips_path = resolve_clips_dir(root_path)

    logger.info("Inspecting dataset at: %s (resolved clips: %s)", root_path, clips_path)

    if not clips_path.exists():
        raise FileNotFoundError(f"Resolved clips directory not found: {clips_path}")

    # Discover classes
    found_dirs = [d for d in clips_path.iterdir() if d.is_dir()]
    found_class_names = {d.name for d in found_dirs}

    missing_target_classes = set(TARGET_CLASSES) - found_class_names
    unexpected_classes = found_class_names - set(TARGET_CLASSES)

    if missing_target_classes:
        logger.warning("Missing target classes: %s", missing_target_classes)
    if unexpected_classes:
        logger.info("Additional / unexpected class directories: %s", unexpected_classes)

    # Collect all video candidate files
    items_to_inspect: List[Tuple[Path, str]] = []
    for d in found_dirs:
        class_name = d.name
        for f in d.iterdir():
            if f.is_file():
                items_to_inspect.append((f, class_name))

    logger.info("Discovered %d candidate files across %d directories.", len(items_to_inspect), len(found_dirs))

    # Parallel inspection
    def _worker(item: Tuple[Path, str]) -> ClipMetadata:
        f_path, c_name = item
        return inspect_single_clip(f_path, c_name, extract_video_info=extract_video_info)

    if max_workers > 1 and len(items_to_inspect) > 20:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            metadata_list = list(executor.map(_worker, items_to_inspect))
    else:
        metadata_list = [_worker(item) for item in items_to_inspect]

    # Compute summary statistics
    class_counts: Dict[str, int] = {}
    format_counts: Dict[str, int] = {}
    readable_count = 0
    unreadable_count = 0
    zero_byte_count = 0
    total_size_bytes = 0
    source_groups: Set[str] = set()

    for item in metadata_list:
        class_counts[item.class_name] = class_counts.get(item.class_name, 0) + 1
        format_counts[item.file_format] = format_counts.get(item.file_format, 0) + 1
        if item.readable:
            readable_count += 1
        else:
            unreadable_count += 1
        if item.file_size_bytes == 0:
            zero_byte_count += 1
        if item.file_size_bytes > 0:
            total_size_bytes += item.file_size_bytes
        if item.source_group:
            source_groups.add(item.source_group)

    summary: Dict[str, Any] = {
        "dataset_root": str(root_path),
        "clips_dir": str(clips_path),
        "total_clips": len(metadata_list),
        "class_counts": class_counts,
        "format_counts": format_counts,
        "readable_clips": readable_count,
        "unreadable_clips": unreadable_count,
        "zero_byte_clips": zero_byte_count,
        "total_size_bytes": total_size_bytes,
        "total_size_mb": round(total_size_bytes / (1024 * 1024), 2),
        "distinct_source_groups": len(source_groups),
        "missing_target_classes": list(missing_target_classes),
        "unexpected_classes": list(unexpected_classes),
    }

    logger.info("Inspection complete: %d clips, %d readable, %d unreadable, %d source groups.",
                len(metadata_list), readable_count, unreadable_count, len(source_groups))

    # Save CSV inventory if requested
    if output_csv:
        out_path = Path(output_csv).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "file_path",
            "filename",
            "class_name",
            "file_format",
            "file_size_bytes",
            "duration_seconds",
            "frame_rate",
            "resolution",
            "frame_count",
            "readable",
            "source_group",
            "error_message",
        ]
        with open(out_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for m in metadata_list:
                writer.writerow(asdict(m))
        logger.info("Saved dataset inventory to: %s", out_path)

    return metadata_list, summary


def main() -> None:
    """CLI entry point for dataset inspection."""
    parser = argparse.ArgumentParser(description="Inspect Badminton Shot Classification Dataset.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="d:/badminton-shot-dataset-lab/data",
        help="Root path to the dataset or data/CLIPS folder.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="Phase -1 Data preparation/report/dataset_inventory.csv",
        help="Destination path for the dataset inventory CSV.",
    )
    parser.add_argument(
        "--no_video_info",
        action="store_true",
        help="Skip deep video decoding and only inspect file system metadata.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Worker thread count for parallel inspection.",
    )

    args = parser.parse_args()

    metadata, summary = run_dataset_inspection(
        dataset_dir=args.dataset_dir,
        output_csv=args.output_csv,
        extract_video_info=not args.no_video_info,
        max_workers=args.max_workers,
    )

    print("\n--- DATASET INSPECTION SUMMARY ---")
    print(f"Total clips found:        {summary['total_clips']}")
    print(f"Total size:               {summary['total_size_mb']} MB")
    print(f"Readable clips:           {summary['readable_clips']}")
    print(f"Unreadable clips:         {summary['unreadable_clips']}")
    print(f"Zero-byte files:          {summary['zero_byte_clips']}")
    print(f"Distinct source groups:   {summary['distinct_source_groups']}")
    print(f"Class distribution:")
    for c, cnt in sorted(summary["class_counts"].items()):
        print(f"  - {c}: {cnt}")
    print(f"File formats:             {summary['format_counts']}")


if __name__ == "__main__":
    main()
