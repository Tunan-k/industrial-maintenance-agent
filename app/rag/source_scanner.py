from __future__ import annotations

import argparse
import hashlib
import json
import uuid

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .manifest import (
    ManifestEntry,
    review_manifest,
    save_manifest,
)


# =========================================================
# 1. Project paths
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


DEFAULT_RAW_ROOT = (
    PROJECT_ROOT
    / "knowledge"
    / "raw"
)


MANIFEST_DIR = (
    PROJECT_ROOT
    / "knowledge"
    / "manifests"
)


SOURCE_META_PATH = (
    MANIFEST_DIR
    / "source_meta.json"
)


MANIFEST_PATH = (
    MANIFEST_DIR
    / "knowledge_sources.jsonl"
)


# =========================================================
# 2. V1 supported formats
# =========================================================

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".md",
    ".txt",
    ".html",
    ".htm",
}


# =========================================================
# 3. SHA256
# =========================================================

def sha256_file(
    path: Path,
    block_size: int = 1024 * 1024,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as file:

        while True:

            chunk = file.read(
                block_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


# =========================================================
# 4. Stable source identity
# =========================================================

def stable_document_id(
    relative_path: str,
) -> str:

    uid = uuid.uuid5(
        uuid.NAMESPACE_URL,
        relative_path,
    )

    return (
        f"doc_{uid.hex[:16]}"
    )


# =========================================================
# 5. Load manually reviewed metadata
# =========================================================

def load_source_meta(
    meta_path: Path,
) -> dict[str, dict[str, Any]]:

    if not meta_path.exists():

        return {}

    raw_text = (
        meta_path
        .read_text(
            encoding="utf-8"
        )
    )

    data = json.loads(
        raw_text
    )

    if not isinstance(
        data,
        dict,
    ):

        raise ValueError(
            "source_meta.json "
            "最外层必须是 JSON object"
        )

    normalized: dict[
        str,
        dict[str, Any],
    ] = {}

    for key, value in data.items():

        if not isinstance(
            value,
            dict,
        ):

            raise ValueError(
                f"{key!r} 对应值 "
                "必须是 JSON object"
            )

        # Windows backslash
        # → internal POSIX-style relative path
        normalized_key = (
            str(key)
            .replace(
                "\\",
                "/",
            )
        )

        normalized[
            normalized_key
        ] = value

    return normalized


# =========================================================
# 6. Skip browser resources / hidden files
# =========================================================

def should_skip(
    path: Path,
    raw_root: Path,
) -> bool:

    relative = (
        path
        .relative_to(
            raw_root
        )
    )

    for part in relative.parts:

        if part.startswith("."):
            return True

        if part.endswith(
            "_files"
        ):
            return True

        if part == "__pycache__":
            return True

    return False


# =========================================================
# 7. Scan knowledge sources
# =========================================================

def scan_sources(
    raw_root: Path,
) -> list[ManifestEntry]:

    raw_root = (
        raw_root
        .expanduser()
        .resolve()
    )

    if not raw_root.exists():

        raise FileNotFoundError(
            "知识源目录不存在："
            f"{raw_root}"
        )

    if not raw_root.is_dir():

        raise NotADirectoryError(
            "知识源路径不是目录："
            f"{raw_root}"
        )

    source_meta = load_source_meta(
        SOURCE_META_PATH
    )

    entries: list[
        ManifestEntry
    ] = []

    files = sorted(
        path
        for path
        in raw_root.rglob("*")
        if path.is_file()
    )

    for path in files:

        if should_skip(
            path,
            raw_root,
        ):
            continue

        suffix = (
            path
            .suffix
            .lower()
        )

        if (
            suffix
            not in SUPPORTED_EXTENSIONS
        ):
            continue

        relative_path = (
            path
            .relative_to(
                raw_root
            )
            .as_posix()
        )

        stat = path.stat()

        override = (
            source_meta
            .get(
                relative_path,
                {},
            )
        )

        entry = ManifestEntry(

            # ---------------------------------------------
            # Automatically observed file facts
            # ---------------------------------------------

            document_id=(
                stable_document_id(
                    relative_path
                )
            ),

            relative_path=(
                relative_path
            ),

            file_type=(
                suffix
                .lstrip(".")
            ),

            file_size=(
                stat.st_size
            ),

            file_sha256=(
                sha256_file(
                    path
                )
            ),

            updated_at=(
                datetime
                .fromtimestamp(
                    stat.st_mtime,
                    tz=timezone.utc,
                )
            ),


            # ---------------------------------------------
            # Human-reviewed industrial metadata
            # ---------------------------------------------

            title=(
                override.get(
                    "title"
                )
            ),

            source=(
                override.get(
                    "source",
                    "unknown",
                )
            ),

            source_url=(
                override.get(
                    "source_url"
                )
            ),

            equipment_type=(
                override.get(
                    "equipment_type",
                    "unknown",
                )
            ),

            equipment_model=(
                override.get(
                    "equipment_model"
                )
            ),

            fault_type=(
                override.get(
                    "fault_type"
                )
            ),

            document_type=(
                override.get(
                    "document_type",
                    "unknown",
                )
            ),

            knowledge_scope=(
                override.get(
                    "knowledge_scope",
                    "unknown",
                )
            ),

            authority_level=(
                override.get(
                    "authority_level",
                    "unknown",
                )
            ),

            language=(
                override.get(
                    "language",
                    "unknown",
                )
            ),

            version=(
                override.get(
                    "version",
                    "1",
                )
            ),

            review_status=(
                override.get(
                    "review_status",
                    "pending",
                )
            ),
        )

        entries.append(
            entry
        )

    return entries


# =========================================================
# 8. CLI
# =========================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "扫描工业知识源并生成 "
            "Knowledge Manifest"
        )
    )

    parser.add_argument(
        "--raw-root",

        type=Path,

        default=DEFAULT_RAW_ROOT,

        help=(
            "原始工业知识源目录。"
            "默认使用 project/knowledge/raw"
        ),
    )

    parser.add_argument(
        "--output",

        type=Path,

        default=MANIFEST_PATH,

        help=(
            "Manifest JSONL 输出路径"
        ),
    )

    args = parser.parse_args()

    raw_root = (
        args.raw_root
        .expanduser()
        .resolve()
    )

    output_path = (
        args.output
        .expanduser()
        .resolve()
    )

    print(
        "[scan] source_root="
        f"{raw_root}"
    )

    entries = scan_sources(
        raw_root=raw_root
    )

    save_manifest(
        entries,
        output_path,
    )

    format_counts = Counter(
        entry.file_type
        for entry in entries
    )

    print(
        "[scan] found="
        f"{len(entries)}"
    )

    print(
        "[scan] formats="
        + json.dumps(
            dict(
                format_counts
            ),
            ensure_ascii=False,
        )
    )

    report = review_manifest(
        entries
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "\nmanifest -> "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()