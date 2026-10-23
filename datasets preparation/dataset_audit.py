"""Dataset Audit Module for Badminton Shot Classification (Phase 1).

Audits structural quality, file integrity, and label consistency across the dataset.
Detects missing, unknown, corrupted, or conflicting labels without modifying
or renaming source data.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
logger = logging.getLogger("dataset_audit")

# Canonical 5 target classes
CANONICAL_CLASSES: Set[str] = {"Clear", "Drive", "Drop", "Net_Shot", "Smash"}

# Normalized alias mapping to canonical names
ALIAS_MAPPING: Dict[str, str] = {
    "clear": "Clear",
    "drive": "Drive",
    "drop": "Drop",
    "smash": "Smash",
    "net_shot": "Net_Shot",
    "net-shot": "Net_Shot",
    "net shot": "Net_Shot",
    "netshot": "Net_Shot",
    "net": "Net_Shot",
}

# Regex to extract label tokens from filenames
FILENAME_LABEL_PATTERN = re.compile(
    r"(?:.*_)?(clear|drive|drop|smash|net_shot|net-shot|net shot|netshot)\.(?:mp4|avi|mov|mkv|webm)$",
    re.IGNORECASE,
)
FB_PREFIX_PATTERN = re.compile(r"^(FB\d{4})_", re.IGNORECASE)
VB_PREFIX_PATTERN = re.compile(r"^VB_.*?_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_", re.IGNORECASE)
AVI_FB_PATTERN = re.compile(r"^(FB_\d{4})_", re.IGNORECASE)


@dataclass
class AuditRecord:
    """Record representing audit outcome for an individual clip."""
    file_path: str
    filename: str
    folder_label: str
    filename_label: str
    canonical_label: str
    is_label_conforming: bool
    label_agreement: bool
    file_size_bytes: int
    readable: bool
    audit_status: str  # VALID, SUSPICIOUS, INVALID
    issue_description: str
    source_group: str


def normalize_label(raw_label: str) -> Optional[str]:
    """Normalizes raw label string against canonical classes and known aliases."""
    if not raw_label:
        return None
    cleaned = raw_label.strip()
    if cleaned in CANONICAL_CLASSES:
        return cleaned
    lookup = cleaned.lower().replace(" ", "_").replace("-", "_")
    if lookup in ALIAS_MAPPING:
        return ALIAS_MAPPING[lookup]
    # Check direct lower match
    if cleaned.lower() in ALIAS_MAPPING:
        return ALIAS_MAPPING[cleaned.lower()]
    return None


def extract_filename_label(filename: str) -> Optional[str]:
    """Extracts shot label token embedded in filename if present."""
    m = FILENAME_LABEL_PATTERN.search(filename)
    if m:
        return normalize_label(m.group(1))
    return None


def extract_source_group(filename: str) -> str:
    """Extracts match or recording session group from filename."""
    m_fb = FB_PREFIX_PATTERN.match(filename)
    if m_fb:
        return m_fb.group(1).upper()
    m_vb = VB_PREFIX_PATTERN.match(filename)
    if m_vb:
        return f"VB_{m_vb.group(1)}"
    m_avi = AVI_FB_PATTERN.match(filename)
    if m_avi:
        return m_avi.group(1).upper()
    parts = filename.split("_")
    return parts[0] if parts else "UNKNOWN"


def audit_clip(
    file_path: Path,
    folder_name: str,
    verify_video_read: bool = False,
) -> AuditRecord:
    """Audits a single clip for label conformance, existence, and integrity."""
    filename = file_path.name
    source_group = extract_source_group(filename)
    norm_folder = normalize_label(folder_name)
    fname_label = extract_filename_label(filename)

    # Check file existence
    if not file_path.exists():
        return AuditRecord(
            file_path=str(file_path),
            filename=filename,
            folder_label=folder_name,
            filename_label=fname_label or "NONE",
            canonical_label=norm_folder or "UNKNOWN",
            is_label_conforming=False,
            label_agreement=False,
            file_size_bytes=-1,
            readable=False,
            audit_status="INVALID",
            issue_description="File not found on filesystem",
            source_group=source_group,
        )

    size = file_path.stat().st_size
    if size == 0:
        return AuditRecord(
            file_path=str(file_path),
            filename=filename,
            folder_label=folder_name,
            filename_label=fname_label or "NONE",
            canonical_label=norm_folder or "UNKNOWN",
            is_label_conforming=False,
            label_agreement=False,
            file_size_bytes=0,
            readable=False,
            audit_status="INVALID",
            issue_description="Empty clip (0 bytes)",
            source_group=source_group,
        )

    # Check label conformity
    if norm_folder is None:
        return AuditRecord(
            file_path=str(file_path),
            filename=filename,
            folder_label=folder_name,
            filename_label=fname_label or "NONE",
            canonical_label="UNKNOWN",
            is_label_conforming=False,
            label_agreement=False,
            file_size_bytes=size,
            readable=True,
            audit_status="INVALID",
            issue_description=f"Folder '{folder_name}' does not map to any of the 5 canonical classes",
            source_group=source_group,
        )

    # Check label agreement between folder and filename
    label_agreement = True
    issues: List[str] = []

    if fname_label is not None and fname_label != norm_folder:
        label_agreement = False
        issues.append(f"Label conflict: folder='{norm_folder}' vs filename='{fname_label}'")

    # Video readability check if requested
    readable = True
    if verify_video_read and HAS_CV2:
        try:
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                readable = False
                issues.append("Failed to open video container")
            else:
                cap.release()
        except Exception as e:
            readable = False
            issues.append(f"OpenCV error: {e}")

    # Determine status
    if not readable:
        status = "INVALID"
    elif not label_agreement:
        status = "SUSPICIOUS"
    else:
        status = "VALID"

    issue_str = "; ".join(issues) if issues else "OK"

    return AuditRecord(
        file_path=str(file_path),
        filename=filename,
        folder_label=folder_name,
        filename_label=fname_label or "NONE",
        canonical_label=norm_folder,
        is_label_conforming=True,
        label_agreement=label_agreement,
        file_size_bytes=size,
        readable=readable,
        audit_status=status,
        issue_description=issue_str,
        source_group=source_group,
    )


def run_dataset_audit(
    dataset_dir: str | Path,
    output_csv: Optional[str | Path] = None,
    verify_video_read: bool = False,
) -> Tuple[List[AuditRecord], Dict[str, Any]]:
    """Performs full audit of dataset labels and structural integrity."""
    root_path = Path(dataset_dir).resolve()
    clips_dir = root_path / "CLIPS" if (root_path / "CLIPS").is_dir() else root_path

    logger.info("Auditing dataset at: %s", clips_dir)
    if not clips_dir.exists():
        raise FileNotFoundError(f"Clips directory not found: {clips_dir}")

    records: List[AuditRecord] = []
    class_subdirs = [d for d in clips_dir.iterdir() if d.is_dir()]

    for cdir in class_subdirs:
        folder_name = cdir.name
        for f in cdir.iterdir():
            if f.is_file():
                record = audit_clip(f, folder_name, verify_video_read=verify_video_read)
                records.append(record)

    # Summarize findings
    status_counts: Dict[str, int] = {"VALID": 0, "SUSPICIOUS": 0, "INVALID": 0}
    canonical_counts: Dict[str, int] = {}
    source_groups: Set[str] = set()

    for r in records:
        status_counts[r.audit_status] = status_counts.get(r.audit_status, 0) + 1
        canonical_counts[r.canonical_label] = canonical_counts.get(r.canonical_label, 0) + 1
        if r.source_group:
            source_groups.add(r.source_group)

    summary: Dict[str, Any] = {
        "total_audited": len(records),
        "status_counts": status_counts,
        "canonical_counts": canonical_counts,
        "valid_count": status_counts.get("VALID", 0),
        "suspicious_count": status_counts.get("SUSPICIOUS", 0),
        "invalid_count": status_counts.get("INVALID", 0),
        "source_groups_count": len(source_groups),
        "audit_note": (
            "Auditing checks structural conformity and folder/filename agreement. "
            "Semantic shot correctness requires verified annotations or visual human review."
        ),
    }

    logger.info("Audit summary: %d clips | VALID=%d, SUSPICIOUS=%d, INVALID=%d",
                len(records), summary["valid_count"], summary["suspicious_count"], summary["invalid_count"])

    if output_csv:
        out_path = Path(output_csv).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "file_path",
            "filename",
            "folder_label",
            "filename_label",
            "canonical_label",
            "is_label_conforming",
            "label_agreement",
            "file_size_bytes",
            "readable",
            "audit_status",
            "issue_description",
            "source_group",
        ]
        with open(out_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(asdict(r))
        logger.info("Audit report written to: %s", out_path)

    return records, summary


def main() -> None:
    """CLI entry point for dataset audit."""
    parser = argparse.ArgumentParser(description="Audit Badminton Dataset Quality and Labels.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="d:/badminton-shot-dataset-lab/data",
        help="Path to dataset directory.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="Phase -1 Data preparation/report/dataset_audit_report.csv",
        help="Path for audit report CSV.",
    )
    parser.add_argument(
        "--verify_video_read",
        action="store_true",
        help="Perform video container opening check with OpenCV.",
    )

    args = parser.parse_args()

    records, summary = run_dataset_audit(
        dataset_dir=args.dataset_dir,
        output_csv=args.output_csv,
        verify_video_read=args.verify_video_read,
    )

    print("\n--- DATASET AUDIT SUMMARY ---")
    print(f"Total clips audited:      {summary['total_audited']}")
    print(f"Valid clips:              {summary['valid_count']}")
    print(f"Suspicious clips:         {summary['suspicious_count']}")
    print(f"Invalid clips:            {summary['invalid_count']}")
    print(f"Canonical label distribution:")
    for c, cnt in sorted(summary["canonical_counts"].items()):
        print(f"  - {c}: {cnt}")
    print(f"\nAudit Note: {summary['audit_note']}")


if __name__ == "__main__":
    main()
