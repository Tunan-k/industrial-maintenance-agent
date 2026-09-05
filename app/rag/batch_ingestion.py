from __future__ import annotations

import argparse
import json

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.rag.ingestion import (
    DocumentIngestionService,
    calculate_file_sha256,
)

from app.rag.manifest import (
    ManifestEntry,
    load_manifest,
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


DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "knowledge"
    / "manifests"
    / "knowledge_sources.jsonl"
)


DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "knowledge"
    / "manifests"
    / "batch_ingestion_report.json"
)


# =========================================================
# 2. Errors
# =========================================================

class BatchIngestionError(RuntimeError):
    """
    Batch ingestion orchestration error.
    """
    pass


# =========================================================
# 3. Resolve source file
# =========================================================

def resolve_source_file(
    raw_root: Path,
    relative_path: str,
) -> Path:

    raw_root = (
        raw_root
        .expanduser()
        .resolve()
    )

    source_file = (
        raw_root
        / relative_path
    ).resolve()

    # 防止 Manifest 中出现 ../ 之类路径逃逸
    try:

        source_file.relative_to(
            raw_root
        )

    except ValueError as exc:

        raise BatchIngestionError(
            "Manifest path escapes "
            "knowledge source root: "
            f"{relative_path}"
        ) from exc

    return source_file


# =========================================================
# 4. Build extra metadata
# =========================================================

def build_extra_metadata(
    entry: ManifestEntry,
) -> dict[str, Any]:

    return {

        "manifest_document_id": (
            entry.document_id
        ),

        "relative_path": (
            entry.relative_path
        ),

        "manifest_file_sha256": (
            entry.file_sha256
        ),

        "source": (
            entry.source
        ),

        "source_url": (
            entry.source_url
        ),

        "fault_type": (
            entry.fault_type
        ),

        "document_type": (
            entry.document_type
        ),

        "knowledge_scope": (
            entry.knowledge_scope
        ),

        # 先保留语义标签
        # 暂时不擅自映射成数值型 authority
        "authority_label": (
            entry.authority_level
        ),

        "review_status": (
            entry.review_status
        ),

        "source_version": (
            entry.version
        ),

        "manifest_language": (
            entry.language
        ),
    }


# =========================================================
# 5. Manufacturer helper
# =========================================================

def resolve_manufacturer(
    entry: ManifestEntry,
) -> str | None:

    # 只有真正被审核为厂家资料时
    # 才把 source 作为 manufacturer
    if (
        entry.authority_level
        == "manufacturer"
    ):

        return entry.source

    return None


# =========================================================
# 6. Ingest one approved Manifest entry
# =========================================================

def ingest_manifest_entry(
    *,
    entry: ManifestEntry,
    raw_root: Path,
    service: DocumentIngestionService,
):

    source_file = (
        resolve_source_file(
            raw_root=raw_root,
            relative_path=(
                entry.relative_path
            ),
        )
    )


    # -----------------------------------------------------
    # 1. File existence
    # -----------------------------------------------------

    if not source_file.exists():

        raise BatchIngestionError(
            "Approved source file "
            "does not exist: "
            f"{source_file}"
        )


    if not source_file.is_file():

        raise BatchIngestionError(
            "Approved source is "
            "not a file: "
            f"{source_file}"
        )


    # -----------------------------------------------------
    # 2. Hash consistency check
    # -----------------------------------------------------

    current_sha256 = (
        calculate_file_sha256(
            source_file
        )
    )


    if (
        current_sha256
        != entry.file_sha256
    ):

        raise BatchIngestionError(
            "Source file changed after "
            "Manifest scan. "
            "Please run Source Scanner "
            "again before ingestion. "
            f"file={entry.relative_path}"
        )


    # -----------------------------------------------------
    # 3. Reuse existing atomic ingestion
    # -----------------------------------------------------

    document = (
        service.ingest_file(

            file_path=(
                source_file
            ),

            title=(
                entry.title
            ),

            equipment_type=(
                entry.equipment_type
            ),

            manufacturer=(
                resolve_manufacturer(
                    entry
                )
            ),

            equipment_model=(
                entry.equipment_model
            ),

            revision=(
                entry.version
            ),

            language=(
                entry.language
            ),

            extra_metadata=(
                build_extra_metadata(
                    entry
                )
            ),

            save_processed=True,
        )
    )

    return document


# =========================================================
# 7. Batch runner
# =========================================================

def run_batch_ingestion(
    *,
    raw_root: Path,
    manifest_path: Path,
    report_path: Path,
) -> dict:

    raw_root = (
        raw_root
        .expanduser()
        .resolve()
    )

    manifest_path = (
        manifest_path
        .expanduser()
        .resolve()
    )

    report_path = (
        report_path
        .expanduser()
        .resolve()
    )


    print(
        "[batch] start"
    )

    print(
        "[batch] raw_root="
        f"{raw_root}"
    )

    print(
        "[batch] manifest="
        f"{manifest_path}"
    )


    # -----------------------------------------------------
    # 1. Load Manifest
    # -----------------------------------------------------

    entries = (
        load_manifest(
            manifest_path
        )
    )


    ready_entries = [
        entry
        for entry in entries
        if entry.ready_for_ingestion()
    ]


    print(
        "[batch] manifest_total="
        f"{len(entries)}"
    )

    print(
        "[batch] ready="
        f"{len(ready_entries)}"
    )


    # -----------------------------------------------------
    # 2. Existing single-file service
    # -----------------------------------------------------

    service = (
        DocumentIngestionService()
    )


    success_count = 0
    failed_count = 0
    skipped_count = 0

    results: list[
        dict[str, Any]
    ] = []


    # -----------------------------------------------------
    # 3. Process entries
    # -----------------------------------------------------

    for entry in entries:

        # pending / rejected 不处理
        if not entry.ready_for_ingestion():

            skipped_count += 1

            continue


        print()

        print(
            "[batch] ingesting: "
            f"{entry.relative_path}"
        )


        try:

            document = (
                ingest_manifest_entry(

                    entry=entry,

                    raw_root=raw_root,

                    service=service,
                )
            )


            entry.parse_status = (
                "success"
            )

            entry.parse_error = None


            success_count += 1


            results.append(
                {
                    "relative_path": (
                        entry.relative_path
                    ),

                    "status": (
                        "success"
                    ),

                    "manifest_document_id": (
                        entry.document_id
                    ),

                    "processed_document_id": (
                        document.document_id
                    ),

                    "title": (
                        document.title
                    ),

                    "source_format": (
                        document.source_format.value
                        if hasattr(
                            document.source_format,
                            "value",
                        )
                        else str(
                            document.source_format
                        )
                    ),

                    "language": (
                        document.language
                    ),
                }
            )


            print(
                "[batch] success: "
                f"{entry.relative_path}"
            )


        except Exception as exc:

            entry.parse_status = (
                "failed"
            )

            entry.parse_error = (
                str(exc)
            )


            failed_count += 1


            results.append(
                {
                    "relative_path": (
                        entry.relative_path
                    ),

                    "status": (
                        "failed"
                    ),

                    "manifest_document_id": (
                        entry.document_id
                    ),

                    "error": (
                        str(exc)
                    ),
                }
            )


            print(
                "[batch] failed: "
                f"{entry.relative_path}"
            )

            print(
                "[batch] error: "
                f"{exc}"
            )


    # -----------------------------------------------------
    # 4. Save updated Manifest
    # -----------------------------------------------------

    save_manifest(
        entries,
        manifest_path,
    )


    # -----------------------------------------------------
    # 5. Batch report
    # -----------------------------------------------------

    report = {

        "generated_at": (
            datetime.now(
                timezone.utc
            )
            .isoformat()
        ),

        "raw_root": (
            str(raw_root)
        ),

        "manifest_path": (
            str(manifest_path)
        ),

        "total_manifest_documents": (
            len(entries)
        ),

        "total_ready": (
            len(ready_entries)
        ),

        "success": (
            success_count
        ),

        "failed": (
            failed_count
        ),

        "skipped": (
            skipped_count
        ),

        "results": (
            results
        ),
    }


    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    report_path.write_text(

        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),

        encoding="utf-8",
    )


    return report


# =========================================================
# 8. CLI
# =========================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Batch ingest approved "
            "industrial knowledge sources"
        )
    )


    parser.add_argument(
        "--raw-root",

        type=Path,

        default=DEFAULT_RAW_ROOT,

        help=(
            "Raw knowledge source root"
        ),
    )


    parser.add_argument(
        "--manifest",

        type=Path,

        default=(
            DEFAULT_MANIFEST_PATH
        ),

        help=(
            "Knowledge Manifest JSONL"
        ),
    )


    parser.add_argument(
        "--report",

        type=Path,

        default=(
            DEFAULT_REPORT_PATH
        ),

        help=(
            "Batch ingestion report path"
        ),
    )


    args = parser.parse_args()


    report = (
        run_batch_ingestion(

            raw_root=(
                args.raw_root
            ),

            manifest_path=(
                args.manifest
            ),

            report_path=(
                args.report
            ),
        )
    )


    print()

    print(
        "[batch] summary"
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )


# =========================================================
# 9. Python module entry
# =========================================================

if __name__ == "__main__":
    main()