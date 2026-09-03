from __future__ import annotations

import argparse
import json

from datetime import datetime
from pathlib import Path
from typing import Any

from app.rag.ingestion import (
    SOURCE_FORMAT_MAP,
    calculate_file_sha256,
)


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

DEFAULT_MANIFEST_DIR = (
    PROJECT_ROOT
    / "knowledge"
    / "manifests"
    / "private"
)

DEFAULT_MANIFEST_PATH = (
    DEFAULT_MANIFEST_DIR
    / "knowledge_sources.local.jsonl"
)


# ============================================================
# Resource-folder rules
# ============================================================

IGNORED_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".vscode",
}


def is_web_resource_folder(
    directory: Path,
) -> bool:
    """
    Browser 'Save webpage complete' often generates:

        page.html
        page_files/

    Files inside *_files are webpage dependencies and
    should not be treated as standalone knowledge sources.
    """

    name = (
        directory
        .name
        .lower()
    )

    return name.endswith(
        "_files"
    )


def should_ignore_directory(
    directory: Path,
) -> bool:

    if directory.name in IGNORED_DIR_NAMES:
        return True

    if is_web_resource_folder(
        directory
    ):
        return True

    return False


# ============================================================
# Source record
# ============================================================

def build_source_record(
    file_path: Path,
    source_root: Path,
) -> dict[str, Any]:
    """
    Build one manifest record.

    Scanner only records objective / technical metadata.
    Business metadata remains pending for later review.
    """

    suffix = (
        file_path
        .suffix
        .lower()
    )

    source_format = (
        SOURCE_FORMAT_MAP[
            suffix
        ]
    )

    content_hash = (
        calculate_file_sha256(
            file_path
        )
    )

    relative_path = (
        file_path
        .relative_to(
            source_root
        )
    )

    return {
        # -------------------------------
        # Stable identity
        # -------------------------------

        "source_id": (
            f"src_{content_hash[:16]}"
        ),

        # -------------------------------
        # Technical metadata
        # -------------------------------

        "source_path": str(
            file_path
        ),

        "relative_path": str(
            relative_path
        ),

        "source_name": (
            file_path.name
        ),

        "source_format": (
            source_format.value
        ),

        "file_size_bytes": (
            file_path
            .stat()
            .st_size
        ),

        "content_sha256": (
            content_hash
        ),

        # -------------------------------
        # Processing state
        # -------------------------------

        "enabled": True,

        "review_status": (
            "pending"
        ),

        "ingestion_status": (
            "not_started"
        ),

        "ingestion_error": None,

        # -------------------------------
        # Business metadata
        #
        # Do NOT guess these only
        # from filename.
        # -------------------------------

        "source_type": (
            "other"
        ),

        "equipment_type": None,

        "manufacturer": None,

        "equipment_model": None,

        "revision": None,

        "authority_level": 3,

        "language": None,

        # -------------------------------
        # Audit information
        # -------------------------------

        "scanned_at": (
            datetime.now()
            .isoformat(
                timespec="seconds"
            )
        ),
    }


# ============================================================
# Directory scanner
# ============================================================
#只负责到底有哪些知识源
def scan_source_directory(
    source_root: str | Path,
) -> list[dict[str, Any]]:
    """
    Recursively discover supported industrial
    knowledge files.

    The function DOES NOT parse document content.
    """

    source_root = (
        Path(source_root)
        .expanduser()
        .resolve()
    )

    if not source_root.exists():

        raise FileNotFoundError(
            f"Source directory does not exist: "
            f"{source_root}"
        )

    if not source_root.is_dir():

        raise NotADirectoryError(
            f"Source root is not a directory: "
            f"{source_root}"
        )


    records: list[
        dict[str, Any]
    ] = []


    for file_path in (
        source_root
        .rglob("*")
    ):

        # --------------------------------
        # Skip directories themselves.
        # --------------------------------

        if not file_path.is_file():
            continue


        # --------------------------------
        # Ignore anything inside
        # webpage *_files directories.
        # --------------------------------

        relative_parts = (
            file_path
            .relative_to(
                source_root
            )
            .parts[:-1]
        )

        ignore_file = False

        for part in relative_parts:

            directory = Path(
                part
            )

            if should_ignore_directory(
                directory
            ):

                ignore_file = True
                break


        if ignore_file:
            continue


        # --------------------------------
        # Skip unsupported extensions.
        # --------------------------------

        suffix = (
            file_path
            .suffix
            .lower()
        )
        # 过滤
        if suffix not in (
            SOURCE_FORMAT_MAP
        ):
            continue


        # --------------------------------
        # Add candidate.
        # --------------------------------

        record = build_source_record(
            file_path=file_path,
            source_root=source_root,
        )

        records.append(
            record
        )


    # Stable ordering improves reproducibility.
    records.sort(
        key=lambda item: (
            item["relative_path"]
            .lower()
        )
    )

    return records


# ============================================================
# Manifest persistence
# ============================================================

def save_manifest(
    records: list[dict[str, Any]],
    manifest_path: str | Path = (
        DEFAULT_MANIFEST_PATH
    ),
) -> Path:

    manifest_path = (
        Path(manifest_path)
        .expanduser()
        .resolve()
    )

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for record in records:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )

            file.write("\n")


    return manifest_path


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Scan industrial knowledge files "
            "and generate a local JSONL manifest."
        )
    )

    parser.add_argument(
        "source_root",
        help=(
            "Root directory containing "
            "industrial knowledge files."
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_MANIFEST_PATH
        ),
        help=(
            "Output JSONL manifest path."
        ),
    )


    args = parser.parse_args()


    records = (
        scan_source_directory(
            args.source_root
        )
    )


    output_path = (
        save_manifest(
            records=records,
            manifest_path=args.output,
        )
    )


    print(
        "===================================="
    )

    print(
        "Industrial Knowledge Source Scanner"
    )

    print(
        "===================================="
    )

    print(
        f"Source root : "
        f"{Path(args.source_root).resolve()}"
    )

    print(
        f"Candidates  : {len(records)}"
    )

    print(
        f"Manifest    : {output_path}"
    )


    format_counts: dict[
        str,
        int,
    ] = {}


    for record in records:

        fmt = record[
            "source_format"
        ]

        format_counts[
            fmt
        ] = (
            format_counts
            .get(
                fmt,
                0,
            )
            + 1
        )


    print(
        "\nFormat summary:"
    )

    for fmt, count in (
        sorted(
            format_counts.items()
        )
    ):

        print(
            f"  {fmt:<10} {count}"
        )


if __name__ == "__main__":
    main()