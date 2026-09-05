"""
功能：它目前只测试 KnowledgeDocument、KnowledgeChunk、SourceLocation、各种 Enum 这些“数据格式定义”是否正确

"""


from app.rag.schemas import (
    ActionType,
    KnowledgeChunk,
    KnowledgeDocument,
    SourceFormat,
    SourceType,
)


def test_knowledge_document():

    document = KnowledgeDocument(
        document_id="doc001",
        title="Drilling Pump Maintenance Manual",
        source_name="pump_manual.pdf",
        source_format=SourceFormat.PDF,
        source_type=SourceType.OEM_MANUAL,
        text="Example maintenance manual content.",
        equipment_type="drilling_pump",
        revision="2025",
        authority_level=5,
    )

    assert document.document_id == "doc001"

    assert (
        document.equipment_type
        == "drilling_pump"
    )

    assert document.authority_level == 5


def test_knowledge_chunk():
    chunk = KnowledgeChunk(
        chunk_id="chunk001",
        document_id="doc001",
        text="Inspect the suction valve and valve seat.",
        metadata={
            "title": "Suction valve inspection",
            "source_name": "pump_manual.pdf",
            "source_format": SourceFormat.PDF,
            "source_type": SourceType.OEM_MANUAL,
            "source_uri": "manuals/pump_manual.pdf",
            "section": "5.3",
            "page": 87,
            "equipment_type": "drilling_pump",
            "components": ["suction_valve"],
            "fault_types": ["suction_severe"],
            "severities": ["severe"],
            "action_types": [ActionType.INSPECTION],
            "authority_level": 5,
        },
    )
    assert chunk.metadata.components == ["suction_valve"]
    assert chunk.metadata.page == 87
    assert chunk.metadata.fault_types == ["suction_severe"]
