"""
Industrial Knowledge Document Ingestion

当前 V1 目标：
    将真实工业知识源统一转换为 KnowledgeDocument。

格式策略：

    TXT / Markdown
        ↓
    Python lightweight reader

    HTML / DOCX / PPTX
        ↓
    Docling
        ↓
    DoclingDocument

    Born-digital PDF
        ↓
    PyMuPDF native text extraction
        ↓
    Page-aware intermediate artifact

最终统一：
    KnowledgeDocument

为什么 PDF 当前不走 Docling：
    当前环境无法访问 Docling PDF layout 模型 Hub。
    即使关闭 OCR、关闭 TableFormer、开启 force_backend_text，
    Docling 仍然初始化 layout predictor，因此继续阻塞 V1。

扫描版 PDF / OCR：
    暂不属于当前 V1 阻塞项，后置处理。
"""

from __future__ import annotations

import hashlib
import json

from pathlib import Path
from typing import Any, Optional, Union

from app.rag.schemas import (
    KnowledgeDocument,
    SourceFormat,
    SourceType,
)


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


DEFAULT_PROCESSED_DIR = (
    PROJECT_ROOT
    / "knowledge"
    / "processed"
)


# ============================================================
# 2. Supported source formats
# ============================================================

SOURCE_FORMAT_MAP = {

    ".pdf": SourceFormat.PDF,

    ".docx": SourceFormat.DOCX,

    ".pptx": SourceFormat.PPTX,

    ".html": SourceFormat.HTML,
    ".htm": SourceFormat.HTML,

    ".txt": SourceFormat.TXT,

    ".md": SourceFormat.MARKDOWN,

    # ingestion 原子层保留图片格式识别能力；
    # 但当前 Source Scanner V1 不主动摄取独立图片。
    ".png": SourceFormat.IMAGE,
    ".jpg": SourceFormat.IMAGE,
    ".jpeg": SourceFormat.IMAGE,
    ".tif": SourceFormat.IMAGE,
    ".tiff": SourceFormat.IMAGE,
    ".bmp": SourceFormat.IMAGE,
    ".webp": SourceFormat.IMAGE,
}


# ============================================================
# 3. Errors
# ============================================================

class IngestionError(RuntimeError):
    """
    Raised when an industrial knowledge source
    cannot be converted into a KnowledgeDocument.
    """
    pass


# ============================================================
# 4. SHA256
# ============================================================

def calculate_file_sha256(
    file_path: Path,
) -> str:
    """
    Incrementally calculate file SHA256.

    用途：
        - 内容指纹
        - 内容版本标识
        - 后续 Change Detection
        - 后续去重 / Upsert 基础
    """

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:

        while True:

            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            sha256.update(block)

    return sha256.hexdigest()


# ============================================================
# 5. Detect source format
# ============================================================

def detect_source_format(
    file_path: Path,
) -> SourceFormat:
    """
    Map file suffix to project SourceFormat.
    """

    suffix = (
        file_path
        .suffix
        .lower()
    )

    source_format = (
        SOURCE_FORMAT_MAP
        .get(suffix)
    )

    if source_format is None:

        raise IngestionError(
            "Unsupported knowledge file format: "
            f"{suffix or '<no extension>'}"
        )

    return source_format


# ============================================================
# 6. TXT / Markdown reader
# ============================================================

def read_text_file(
    file_path: Path,
) -> str:
    """
    Read TXT / Markdown using common
    Chinese and English encodings.
    """

    last_error = None

    for encoding in (
        "utf-8-sig",
        "utf-8",
        "gb18030",
        "gbk",
    ):

        try:

            return file_path.read_text(
                encoding=encoding
            )

        except UnicodeDecodeError as exc:

            last_error = exc

    raise IngestionError(
        "Unable to decode text file: "
        f"{file_path.name}. "
        f"Last error: {last_error}"
    )


# ============================================================
# 7. Document Ingestion Service
# ============================================================

class DocumentIngestionService:
    """
    Convert heterogeneous industrial knowledge files
    into unified KnowledgeDocument.

    当前 V1 Parser 路由：

        TXT / MD
            → builtin reader

        PDF
            → PyMuPDF native text

        HTML / DOCX / PPTX / Image
            → Docling

    Batch Ingestion 不关心这些 Parser 细节，
    只重复调用 ingest_file()。
    """

    def __init__(
        self,
        processed_dir: Union[
            str,
            Path,
        ] = DEFAULT_PROCESSED_DIR,
    ):

        self.processed_dir = Path(
            processed_dir
        )

        # Unified KnowledgeDocument
        self.document_dir = (
            self.processed_dir
            / "documents"
        )

        # Rich Docling structural artifacts
        self.docling_dir = (
            self.processed_dir
            / "docling"
        )

        # Born-digital PDF page-aware artifacts
        self.pdf_pages_dir = (
            self.processed_dir
            / "pdf_pages"
        )

        self.document_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.docling_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.pdf_pages_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Lazy loading:
        # HTML / DOCX / PPTX 真正需要时
        # 才初始化 Docling。
        self._converter = None


    # ========================================================
    # 8. Lazy Docling loader
    # ========================================================

    def _get_converter(self):
        """
        Docling is currently used for:

            HTML
            DOCX
            PPTX
            Image

        Born-digital PDF does NOT enter this path
        in current V1.
        """

        if self._converter is None:

            try:

                from docling.document_converter import (
                    DocumentConverter,
                )

            except ImportError as exc:

                raise IngestionError(
                    "Docling is required for "
                    "HTML/DOCX/PPTX/Image ingestion."
                ) from exc

            self._converter = (
                DocumentConverter()
            )

        return self._converter


    # ========================================================
    # 9. Born-digital PDF native text reader
    # ========================================================

    def _read_born_digital_pdf(
        self,
        *,
        file_path: Path,
        document_id: str,
        save_processed: bool,
    ) -> tuple[
        str,
        int,
        Optional[Path],
    ]:
        """
        Extract text directly from the PDF text layer.

        当前适用范围：
            - 学术论文
            - 数字技术手册
            - 可选择/复制文字的数字 PDF

        当前不负责：
            - 扫描件
            - OCR
            - 图片文字识别

        同时保存 page-aware JSON：

            page_number
            text

        为下一阶段 Chunking 保留页级信息。
        """

        try:

            import fitz

        except ImportError as exc:

            raise IngestionError(
                "PyMuPDF is required for "
                "born-digital PDF ingestion. "
                "Install it with: pip install PyMuPDF"
            ) from exc


        pages: list[
            dict[str, Any]
        ] = []

        text_parts: list[str] = []

        pdf_pages_path: Optional[
            Path
        ] = None


        try:

            pdf_document = (
                fitz.open(
                    file_path
                )
            )

        except Exception as exc:

            raise IngestionError(
                "PyMuPDF failed to open PDF "
                f"{file_path.name}: {exc}"
            ) from exc


        try:

            page_count = len(
                pdf_document
            )


            for page_index in range(
                page_count
            ):

                page = (
                    pdf_document[
                        page_index
                    ]
                )


                # sort=True:
                # 尽量依据页面坐标重建阅读顺序。
                #
                # 对复杂双栏页面不保证完美，
                # 因此后面必须做 PDF Pilot 内容验收。
                page_text = (
                    page
                    .get_text(
                        "text",
                        sort=True,
                    )
                    .strip()
                )


                page_number = (
                    page_index + 1
                )


                pages.append(
                    {
                        "page_number": (
                            page_number
                        ),

                        "text": (
                            page_text
                        ),
                    }
                )


                if page_text:

                    text_parts.append(
                        (
                            f"\n\n"
                            f"<!-- page:{page_number} -->"
                            f"\n\n"
                            f"{page_text}"
                        )
                    )


            normalized_text = (
                "\n".join(
                    text_parts
                )
                .strip()
            )


            if save_processed:

                pdf_pages_path = (
                    self.pdf_pages_dir
                    / (
                        f"{document_id}"
                        ".pages.json"
                    )
                )


                page_artifact = {

                    "document_id": (
                        document_id
                    ),

                    "source_name": (
                        file_path.name
                    ),

                    "parser": (
                        "pymupdf_native_text"
                    ),

                    "page_count": (
                        page_count
                    ),

                    "pages": (
                        pages
                    ),
                }


                pdf_pages_path.write_text(

                    json.dumps(
                        page_artifact,
                        ensure_ascii=False,
                        indent=2,
                    ),

                    encoding="utf-8",
                )


            return (
                normalized_text,
                page_count,
                pdf_pages_path,
            )


        finally:

            pdf_document.close()


    # ========================================================
    # 10. Main public API
    # ========================================================

    def ingest_file(
        self,
        file_path: Union[
            str,
            Path,
        ],
        *,
        title: Optional[str] = None,
        source_type: SourceType = (
            SourceType.OTHER
        ),
        equipment_type: Optional[str] = None,
        manufacturer: Optional[str] = None,
        equipment_model: Optional[str] = None,
        revision: Optional[str] = None,
        authority_level: int = 3,
        language: str = "zh",
        extra_metadata: Optional[
            dict[str, Any]
        ] = None,
        save_processed: bool = True,
    ) -> KnowledgeDocument:
        """
        Ingest exactly one industrial knowledge file.

        Stable atomic operation:

            one source file
                ↓
            parser
                ↓
            normalized text
                ↓
            KnowledgeDocument

        Batch Ingestion repeatedly calls this method.
        """


        # ----------------------------------------------------
        # 1. Normalize path
        # ----------------------------------------------------

        file_path = (
            Path(file_path)
            .expanduser()
            .resolve()
        )


        # ----------------------------------------------------
        # 2. Validate input
        # ----------------------------------------------------

        if not file_path.exists():

            raise IngestionError(
                "Knowledge file does not exist: "
                f"{file_path}"
            )


        if not file_path.is_file():

            raise IngestionError(
                "Knowledge source must be a file: "
                f"{file_path}"
            )


        # ----------------------------------------------------
        # 3. Detect source format
        # ----------------------------------------------------

        source_format = (
            detect_source_format(
                file_path
            )
        )


        # ----------------------------------------------------
        # 4. Stable processed-document identity
        # ----------------------------------------------------

        content_hash = (
            calculate_file_sha256(
                file_path
            )
        )

        document_id = (
            f"doc_{content_hash[:16]}"
        )


        # ----------------------------------------------------
        # 5. Parser output placeholders
        # ----------------------------------------------------

        parser_name = None

        page_count = None

        docling_json_path = None

        pdf_pages_path = None


        # ====================================================
        # A. TXT / Markdown
        # ====================================================

        if source_format in {
            SourceFormat.TXT,
            SourceFormat.MARKDOWN,
        }:

            normalized_text = (
                read_text_file(
                    file_path
                )
            )

            parser_name = (
                "builtin_text_reader"
            )


        # ====================================================
        # B. Born-digital PDF
        # ====================================================

        elif (
            source_format
            == SourceFormat.PDF
        ):

            (
                normalized_text,
                page_count,
                pdf_pages_path,
            ) = self._read_born_digital_pdf(

                file_path=(
                    file_path
                ),

                document_id=(
                    document_id
                ),

                save_processed=(
                    save_processed
                ),
            )

            parser_name = (
                "pymupdf_native_text"
            )


        # ====================================================
        # C. Docling formats
        # ====================================================

        else:

            try:

                converter = (
                    self._get_converter()
                )

                result = (
                    converter.convert(
                        file_path
                    )
                )

                docling_document = (
                    result.document
                )


                # --------------------------------------------
                # Readable Markdown representation
                # --------------------------------------------

                normalized_text = (
                    docling_document
                    .export_to_markdown()
                )

                parser_name = (
                    "docling"
                )


                # --------------------------------------------
                # Page count if available
                # --------------------------------------------

                try:

                    pages = getattr(
                        docling_document,
                        "pages",
                        None,
                    )

                    if pages is not None:

                        page_count = len(
                            pages
                        )

                except Exception:

                    page_count = None


                # --------------------------------------------
                # Preserve Docling structure
                # --------------------------------------------

                if save_processed:

                    from docling_core.types.doc import (
                        ImageRefMode,
                    )

                    docling_json_path = (
                        self.docling_dir
                        / (
                            f"{document_id}"
                            ".docling.json"
                        )
                    )

                    docling_document.save_as_json(
                        docling_json_path,
                        image_mode=(
                            ImageRefMode.PLACEHOLDER
                        ),
                    )


            except IngestionError:

                raise


            except Exception as exc:

                raise IngestionError(
                    "Docling failed to parse "
                    f"{file_path.name}: {exc}"
                ) from exc


        # ----------------------------------------------------
        # 6. Validate normalized content
        # ----------------------------------------------------

        normalized_text = (
            normalized_text
            .strip()
        )

        if not normalized_text:

            if (
                source_format
                == SourceFormat.PDF
            ):

                raise IngestionError(
                    "No usable embedded text "
                    f"was extracted from {file_path.name}. "
                    "This PDF may be scanned/image-based. "
                    "OCR is not enabled in current V1."
                )

            raise IngestionError(
                "No usable text was extracted "
                f"from {file_path.name}."
            )


        # ----------------------------------------------------
        # 7. Technical metadata
        # ----------------------------------------------------

        metadata = {

            "content_sha256": (
                content_hash
            ),

            "file_size_bytes": (
                file_path
                .stat()
                .st_size
            ),

            "parser": (
                parser_name
            ),

            "page_count": (
                page_count
            ),

            "docling_json_path": (
                str(
                    docling_json_path
                )
                if docling_json_path
                else None
            ),

            "pdf_pages_path": (
                str(
                    pdf_pages_path
                )
                if pdf_pages_path
                else None
            ),

            "text_representation": (
                "markdown"
                if parser_name
                == "docling"

                else (
                    "page_text"
                    if parser_name
                    == "pymupdf_native_text"

                    else
                    "plain_text"
                )
            ),

            "pdf_ocr_enabled": (
                False
                if source_format
                == SourceFormat.PDF
                else None
            ),

            "pdf_parser_strategy": (
                "born_digital_native_text"
                if source_format
                == SourceFormat.PDF
                else None
            ),
        }


        # ----------------------------------------------------
        # 8. Merge Manifest / Batch metadata
        # ----------------------------------------------------

        if extra_metadata:

            metadata.update(
                extra_metadata
            )


        # ----------------------------------------------------
        # 9. Build unified KnowledgeDocument
        # ----------------------------------------------------

        document = KnowledgeDocument(

            document_id=(
                document_id
            ),

            title=(
                title
                or file_path.stem
            ),

            source_name=(
                file_path.name
            ),

            source_format=(
                source_format
            ),

            source_type=(
                source_type
            ),

            source_uri=str(
                file_path
            ),

            text=(
                normalized_text
            ),

            language=(
                language
            ),

            equipment_type=(
                equipment_type
            ),

            manufacturer=(
                manufacturer
            ),

            equipment_model=(
                equipment_model
            ),

            revision=(
                revision
            ),

            authority_level=(
                authority_level
            ),

            metadata=(
                metadata
            ),
        )


        # ----------------------------------------------------
        # 10. Persist unified KnowledgeDocument
        # ----------------------------------------------------

        if save_processed:

            knowledge_json_path = (
                self.document_dir
                / (
                    f"{document_id}"
                    ".knowledge.json"
                )
            )

            knowledge_json_path.write_text(
                document.model_dump_json(
                    indent=2
                ),
                encoding="utf-8",
            )


        # ----------------------------------------------------
        # 11. Return
        # ----------------------------------------------------

        return document