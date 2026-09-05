"""Conservative source classification; document meaning precedes file format."""
from app.rag.schemas import SourceType


def resolve_source_type(document_type: str | None, source_format: str) -> SourceType:
    known = {
        "research_paper": SourceType.RESEARCH_PAPER,
        "industry_safety_guidance": SourceType.INDUSTRY_SAFETY_GUIDANCE,
        "internal_project_note": SourceType.INTERNAL_PROJECT_NOTE,
    }
    if document_type in known:
        return known[document_type]
    try:
        parsed = SourceType(document_type)
        if parsed != SourceType.OTHER:
            return parsed
    except (ValueError, TypeError):
        pass
    return SourceType.PPT if source_format.lower().lstrip(".") in {"ppt", "pptx"} else SourceType.OTHER
