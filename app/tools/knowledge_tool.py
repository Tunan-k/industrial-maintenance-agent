"""Framework-free Knowledge Tool. The caller owns the injected Retriever lifecycle."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.rag.embedding import EmbeddingError
from app.rag.retriever import Retriever
from app.rag.schemas import Evidence
from app.rag.vector_store import VectorStoreError


class KnowledgeToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    query: str = Field(min_length=1)
    equipment_type: str | None = Field(default=None, min_length=1)
    top_k: int = Field(default=5, gt=0)


class KnowledgeToolError(RuntimeError):
    """Stable error code and safe message; underlying exception remains __cause__."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class KnowledgeTool:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def retrieve_knowledge(self, query: str, equipment_type: str | None = None,
                           top_k: int = 5) -> list[Evidence]:
        try:
            request = KnowledgeToolInput(query=query, equipment_type=equipment_type, top_k=top_k)
        except ValidationError as exc:
            raise KnowledgeToolError("invalid_input", "Invalid knowledge query, equipment type or top_k.") from exc
        filters = {"equipment_type": request.equipment_type} if request.equipment_type is not None else None
        try:
            evidence = self.retriever.retrieve(request.query, top_k=request.top_k, filters=filters)
        except EmbeddingError as exc:
            raise KnowledgeToolError("embedding_error", "Knowledge query encoding failed.") from exc
        except VectorStoreError as exc:
            raise KnowledgeToolError("vector_store_error", "Knowledge vector search failed.") from exc
        except Exception as exc:
            raise KnowledgeToolError("retrieval_error", "Knowledge retrieval failed.") from exc
        # Keep the existing Evidence contract. Failures must never look like [];
        # only a successful empty retrieval is an empty result.
        return evidence


_configured_tool: KnowledgeTool | None = None


def configure_knowledge_tool(retriever: Retriever | None) -> None:
    """Bind once during application setup; None unbinds without closing caller resources.

    Multiple independent configurations should use KnowledgeTool instances instead.
    """
    global _configured_tool
    _configured_tool = KnowledgeTool(retriever) if retriever is not None else None


def retrieve_knowledge(query: str, equipment_type: str | None = None,
                       top_k: int = 5) -> list[Evidence]:
    """Return source Evidence from the explicitly configured existing Retriever."""
    tool = _configured_tool
    if tool is None:
        raise KnowledgeToolError("not_configured", "Configure a Retriever before calling the Knowledge Tool.")
    return tool.retrieve_knowledge(query, equipment_type, top_k)
