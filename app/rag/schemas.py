from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# Enumerations:把字段取值标准化
# ============================================================


class SourceFormat(str, Enum):
    """
    Original file / source format.
    """

    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    HTML = "html"
    TXT = "txt"
    IMAGE = "image"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


class SourceType(str, Enum):
    """
    Business meaning / authority category of a knowledge source.
    """

    OEM_MANUAL = "oem_manual"
    TROUBLESHOOTING = "troubleshooting"
    MAINTENANCE_SOP = "maintenance_sop"
    SAFETY_PROCEDURE = "safety_procedure"
    INSPECTION_STANDARD = "inspection_standard"
    INDUSTRY_STANDARD = "industry_standard"
    REPAIR_CASE = "repair_case"
    PARTS_MANUAL = "parts_manual"
    TECHNICAL_REPORT = "technical_report"
    RESEARCH_PAPER = "research_paper"
    OTHER = "other"


class ContentType(str, Enum):
    """
    Semantic form of the chunk content.
    """

    TEXT = "text"
    TABLE = "table"
    PROCEDURE = "procedure"
    WARNING = "warning"
    LIST = "list"
    FIGURE_CAPTION = "figure_caption"
    OTHER = "other"


class ActionType(str, Enum):
    """
    Maintenance action represented by a knowledge chunk.
    """

    DIAGNOSIS = "diagnosis"
    INSPECTION = "inspection"
    REPAIR = "repair"
    REPLACEMENT = "replacement"
    ADJUSTMENT = "adjustment"
    SAFETY = "safety"
    PREVENTIVE_MAINTENANCE = "preventive_maintenance"
    TROUBLESHOOTING = "troubleshooting"
    OTHER = "other"


# ============================================================
# Source location
# ============================================================


class SourceLocation(BaseModel):
    """
    Position of a chunk in the original document.
    """

    section: Optional[str] = Field(
        default=None,
        description="Section or heading containing the content.",
    )

    page: Optional[int] = Field(
        default=None,
        ge=1,
        description="PDF or document page number.",
    )

    slide: Optional[int] = Field(
        default=None,
        ge=1,
        description="PowerPoint slide number.",
    )

    element_index: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Element sequence number inside the parsed document."
        ),
    )


# ============================================================
# Document-level schema
# ============================================================


class KnowledgeDocument(BaseModel):
    """
    Unified representation of one parsed industrial document.

    A PDF, DOCX, PPTX, HTML, TXT or image should eventually
    be normalized into this representation before chunking.
    """

    document_id: str = Field(
        min_length=1,
        description=(
            "Stable unique identifier of the document."
        ),
    )

    title: str = Field(
        min_length=1,
        description="Human-readable document title.",
    )

    source_name: str = Field(
        min_length=1,
        description=(
            "Original filename or source display name."
        ),
    )

    source_format: SourceFormat = Field(
        description="Original source format.",
    )

    source_type: SourceType = Field(
        default=SourceType.OTHER,
        description=(
            "Business category of the industrial knowledge source."
        ),
    )

    source_uri: Optional[str] = Field(
        default=None,
        description=(
            "Original local path, URL or external source reference."
        ),
    )

    text: str = Field(
        default="",
        description=(
            "Normalized text extracted from the source document."
        ),
    )

    language: str = Field(
        default="zh",
        description=(
            "Document language, e.g. zh or en."
        ),
    )

    equipment_type: Optional[str] = Field(
        default=None,
        description=(
            "Equipment type, e.g. drilling_pump."
        ),
    )

    manufacturer: Optional[str] = Field(
        default=None,
        description="Equipment manufacturer.",
    )

    equipment_model: Optional[str] = Field(
        default=None,
        description="Equipment model or product series.",
    )

    revision: Optional[str] = Field(
        default=None,
        description=(
            "Document revision, version or edition."
        ),
    )

    authority_level: int = Field(
        default=3,
        ge=1,
        le=5,
        description=(
            "Knowledge authority level. "
            "5 is highest authority."
        ),
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional parser- or domain-specific metadata."
        ),
    )


# ============================================================
# Retrieval-level schema
# ============================================================


class ChunkMetadata(BaseModel):
    """Shared provenance and indexing metadata for every source format.

    Nullable required fields distinguish unknown provenance from omitted keys.
    Numeric authority is inherited from KnowledgeDocument; authority_label is
    preserved separately without inventing a numeric ranking for source labels.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    source_type: SourceType
    source_uri: Optional[str]
    equipment_type: Optional[str]
    authority_level: int = Field(ge=1, le=5)
    page: Optional[int] = Field(ge=1)
    section: Optional[str]

    source_name: str = Field(min_length=1)
    source_format: SourceFormat
    language: str = "zh"
    equipment_model: Optional[str] = None
    revision: Optional[str] = None
    slide: Optional[int] = Field(default=None, ge=1)
    manifest_document_id: str = ""
    chunk_index: int = Field(default=0, ge=0)
    source: Optional[str] = None
    source_url: Optional[str] = None
    document_type: Optional[str] = None
    knowledge_scope: Optional[str] = None
    authority_label: Optional[str] = None
    relative_path: Optional[str] = None
    fault_type: Optional[str] = None
    content_type: ContentType = ContentType.TEXT
    components: list[str] = Field(default_factory=list)
    fault_types: list[str] = Field(default_factory=list)
    severities: list[str] = Field(default_factory=list)
    action_types: list[ActionType] = Field(default_factory=list)
    chunking_strategy: Optional[str] = None
    char_count: Optional[int] = Field(default=None, ge=0)
    estimated_tokens: Optional[int] = Field(default=None, ge=0)


class KnowledgeChunk(BaseModel):
    """Canonical serialized chunk contract; provenance lives only in metadata."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    metadata: ChunkMetadata


class Evidence(BaseModel):
    """Retrieved source evidence; score is similarity, not fault probability."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    score: float = Field(allow_inf_nan=False)
    metadata: ChunkMetadata
