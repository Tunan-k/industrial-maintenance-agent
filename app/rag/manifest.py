from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel


# =========================================================
# 1. Manifest data contract
# =========================================================

class ManifestEntry(BaseModel):
    """
    One ManifestEntry represents one raw industrial
    knowledge source.

    Manifest contains three categories of information:

    1. File facts
       Generated automatically by Source Scanner.

    2. Industrial business metadata
       Reviewed manually through source_meta.json.

    3. Pipeline status
       Human review + parsing status.
    """

    # =====================================================
    # A. File identity
    # =====================================================

    document_id: str

    # Path relative to the knowledge source root
    relative_path: str

    # pdf / html / docx / pptx / ...
    file_type: str

    # File size in bytes
    file_size: int

    # SHA256 of raw file content
    file_sha256: str

    # Last modified time
    updated_at: datetime


    # =====================================================
    # B. Industrial metadata
    # =====================================================

    title: Optional[str] = None

    # Human-readable source organization / source name
    source: str = "unknown"

    # Original URL when available
    source_url: Optional[str] = None

    # drilling_pump / bearing / gearbox / ...
    equipment_type: str = "unknown"

    # HH2400 etc.
    equipment_model: Optional[str] = None

    # Optional because generic documents may cover
    # multiple failure modes
    fault_type: Optional[str] = None

    # research_paper / manufacturer_manual /
    # industry_safety_guidance / standard / ...
    document_type: str = "unknown"

    # model_specific
    # equipment_type
    # component
    # general_industry
    knowledge_scope: str = "unknown"

    # manufacturer
    # industry_association
    # official_standard
    # research_reference
    # textbook
    authority_level: str = "unknown"

    # zh / en
    language: str = "unknown"

    version: str = "1"


    # =====================================================
    # C. Human review
    # =====================================================

    review_status: Literal[
        "pending",
        "approved",
        "rejected",
    ] = "pending"


    # =====================================================
    # D. Parsing status
    # =====================================================

    parse_status: Literal[
        "pending",
        "success",
        "failed",
        "skipped",
    ] = "pending"

    parse_error: Optional[str] = None


    # =====================================================
    # 2. Metadata completeness
    # =====================================================

    def metadata_ready(self) -> bool:
        """
        equipment_model and fault_type are NOT mandatory.

        Generic drilling-pump knowledge may not belong
        to one specific model or one specific fault.
        """

        return all(
            [
                self.source not in ("", "unknown"),
                self.equipment_type not in ("", "unknown"),
                self.document_type not in ("", "unknown"),
                self.knowledge_scope not in ("", "unknown"),
                self.authority_level not in ("", "unknown"),
                self.language not in ("", "unknown"),
            ]
        )


    # =====================================================
    # 3. Ingestion gate
    # =====================================================

    def ready_for_ingestion(self) -> bool:
        """
        A document is allowed to enter Batch Ingestion only if:

        1. Industrial metadata is complete enough.
        2. Human review explicitly approved it.
        """

        metadata_ok = self.metadata_ready()

        human_approved = (
            self.review_status == "approved"
        )

        return (
            metadata_ok
            and human_approved
        )


# =========================================================
# 4. Save Manifest
# =========================================================

def save_manifest(
    entries: list[ManifestEntry],
    manifest_path: Path,
) -> None:

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for entry in entries:

            file.write(
                entry.model_dump_json()
                + "\n"
            )


# =========================================================
# 5. Load Manifest
# =========================================================

def load_manifest(
    manifest_path: Path,
) -> list[ManifestEntry]:

    if not manifest_path.exists():

        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    entries: list[ManifestEntry] = []

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                entry = (
                    ManifestEntry
                    .model_validate_json(line)
                )

                entries.append(entry)

            except Exception as exc:

                raise ValueError(
                    "Invalid manifest "
                    f"line {line_number}: {exc}"
                ) from exc

    return entries


# =========================================================
# 6. Manifest Review
# =========================================================

def review_manifest(
    entries: list[ManifestEntry],
) -> dict:

    ready: list[ManifestEntry] = []

    needs_review: list[ManifestEntry] = []

    rejected: list[ManifestEntry] = []

    for entry in entries:

        if entry.review_status == "rejected":

            rejected.append(entry)

        elif entry.ready_for_ingestion():

            ready.append(entry)

        else:

            needs_review.append(entry)

    return {

        "total": len(entries),

        "ready": len(ready),

        "needs_review": len(
            needs_review
        ),

        "rejected": len(
            rejected
        ),

        "ready_documents": [
            entry.relative_path
            for entry in ready
        ],

        "needs_review_documents": [
            entry.relative_path
            for entry in needs_review
        ],

        "rejected_documents": [
            entry.relative_path
            for entry in rejected
        ],
    }