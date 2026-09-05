from pathlib import Path

from app.rag.source_scanner import (
    scan_sources,
)


def test_source_scanner_ignores_web_resources(
    tmp_path: Path,
):

    # ------------------------------------
    # Real knowledge files
    # ------------------------------------

    html_file = (
        tmp_path
        / "manual.html"
    )

    html_file.write_text(
        "<html>manual</html>",
        encoding="utf-8",
    )


    pdf_file = (
        tmp_path
        / "manual.pdf"
    )

    pdf_file.write_bytes(
        b"fake-pdf"
    )


    # ------------------------------------
    # Browser resource folder
    # ------------------------------------

    resource_dir = (
        tmp_path
        / "manual_files"
    )

    resource_dir.mkdir()


    resource_image = (
        resource_dir
        / "logo.png"
    )

    resource_image.write_bytes(
        b"fake-image"
    )


    # ------------------------------------
    # Scan
    # ------------------------------------

    records = (
        scan_sources(
            tmp_path
        )
    )


    names = {
        record.relative_path
        for record in records
    }


    assert (
        "manual.html"
        in names
    )

    assert (
        "manual.pdf"
        in names
    )

    assert (
        "logo.png"
        not in names
    )