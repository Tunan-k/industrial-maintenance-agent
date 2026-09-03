from pathlib import Path

import pytest

from app.rag.ingestion import (
    DocumentIngestionService,
    IngestionError,
)

from app.rag.schemas import (
    SourceFormat,
    SourceType,
)


def test_txt_ingestion(
    tmp_path: Path,
):

    source_file = (
        tmp_path
        / "demo.txt"
    )

    source_file.write_text(
        (
            "SOFTWARE TEST DOCUMENT ONLY\n\n"
            "Equipment: drilling_pump\n"
            "Component: suction_valve\n"
        ),
        encoding="utf-8",
    )


    processed_dir = (
        tmp_path
        / "processed"
    )


    service = (
        DocumentIngestionService(
            processed_dir=processed_dir
        )
    )


    document = service.ingest_file(
        source_file,

        source_type=(
            SourceType.OTHER
        ),

        equipment_type=(
            "drilling_pump"
        ),

        authority_level=1,
    )


    assert (
        document.source_format
        == SourceFormat.TXT
    )

    assert (
        document.equipment_type
        == "drilling_pump"
    )

    assert (
        "SOFTWARE TEST DOCUMENT ONLY"
        in document.text
    )

    assert (
        document.metadata[
            "parser"
        ]
        == "builtin_text_reader"
    )

    assert (
        document.document_id
        .startswith("doc_")
    )


def test_ingestion_persists_document(
    tmp_path: Path,
):

    source_file = (
        tmp_path
        / "demo.txt"
    )

    source_file.write_text(
        "test ingestion content",
        encoding="utf-8",
    )


    processed_dir = (
        tmp_path
        / "processed"
    )


    service = (
        DocumentIngestionService(
            processed_dir=processed_dir
        )
    )


    document = service.ingest_file(
        source_file,
        authority_level=1,
    )


    expected_file = (
        processed_dir
        / "documents"
        / (
            f"{document.document_id}"
            ".knowledge.json"
        )
    )


    assert expected_file.exists()

def test_markdown_ingestion(
    tmp_path: Path,
):

    source_file = (
        tmp_path
        / "demo.md"
    )

    source_file.write_text(
        "# Software Test\n\n"
        "This is a markdown ingestion test.",
        encoding="utf-8",
    )

    service = DocumentIngestionService(
        processed_dir=(
            tmp_path
            / "processed"
        )
    )

    document = service.ingest_file(
        source_file,
        authority_level=1,
    )

    assert (
        document.source_format
        == SourceFormat.MARKDOWN
    )

    assert (
        document.metadata["parser"]
        == "builtin_text_reader"
    )

    assert (
        "Software Test"
        in document.text
    )


def test_unsupported_format(
    tmp_path: Path,
):

    source_file = (
        tmp_path
        / "demo.xyz"
    )

    source_file.write_text(
        "test",
        encoding="utf-8",
    )


    service = (
        DocumentIngestionService(
            processed_dir=(
                tmp_path
                / "processed"
            )
        )
    )


    with pytest.raises(
        IngestionError
    ):

        service.ingest_file(
            source_file
        )

