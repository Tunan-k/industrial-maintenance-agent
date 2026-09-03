"""
功能：
    ① 校验文件 file_path.exists()
    ② 判断原始格式 PDF / DOCX / PPTX / HTML / TXT / Image
    ③ Parser TXT → Python读取；其他 → Docling
    ④ 得到统一文本，normalized_text
    ⑤ 构造KnowledgeDocument(...)
    ⑥ 保存 processed artifact

代码详解：
hashlib：这是 Python 标准库，主要用于：计算文件 SHA256
calculate_file_sha256(...)作用之一就是给文件做指纹

"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional, Union

from app.rag.schemas import (
    KnowledgeDocument,
    SourceFormat,
    SourceType,
)


# ============================================================
# Project paths
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
# Supported source formats
# ============================================================

SOURCE_FORMAT_MAP = {
    ".pdf": SourceFormat.PDF,

    ".docx": SourceFormat.DOCX,

    ".pptx": SourceFormat.PPTX,

    ".html": SourceFormat.HTML,
    ".htm": SourceFormat.HTML,

    ".txt": SourceFormat.TXT,

    ".md": SourceFormat.MARKDOWN,

    ".png": SourceFormat.IMAGE,
    ".jpg": SourceFormat.IMAGE,
    ".jpeg": SourceFormat.IMAGE,
    ".tif": SourceFormat.IMAGE,
    ".tiff": SourceFormat.IMAGE,
    ".bmp": SourceFormat.IMAGE,
    ".webp": SourceFormat.IMAGE,
}


# ============================================================
# Errors
# ============================================================

class IngestionError(RuntimeError):
    """
    Raised when an industrial knowledge source
    cannot be converted into a KnowledgeDocument.
    """


# ============================================================
# Helper functions
# ============================================================

#文件内容↓SHA256↓稳定 document_id
def calculate_file_sha256(
    file_path: Path,
) -> str:
    """
    Calculate SHA256 incrementally.

    The whole file is not loaded into memory at once.
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


def detect_source_format(
    file_path: Path,
) -> SourceFormat:
    """
    Map a file extension to the project's SourceFormat.
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


def read_text_file(
    file_path: Path,
) -> str:
    """
    Read TXT / Markdown using common Chinese
    and English encodings.
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
# Document Ingestion Service
# ============================================================

class DocumentIngestionService:
    """
    Convert heterogeneous industrial knowledge files
    into the project's unified KnowledgeDocument.

    Pipeline:

        TXT / Markdown
              ↓
        lightweight reader
              ↓
        KnowledgeDocument


        PDF / DOCX / PPTX / HTML / Image
              ↓
            Docling
              ↓
        DoclingDocument
              ↓
        KnowledgeDocument
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

        self.document_dir = (
            self.processed_dir
            / "documents"
        )

        self.docling_dir = (
            self.processed_dir
            / "docling"
        )

        self.document_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.docling_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Lazy loading:
        #
        # Do not create Docling's DocumentConverter
        # until a complex document really needs it.
        # ----------------------------------------------------

        self._converter = None


    # ========================================================
    # Lazy Docling loader
    # ========================================================

    def _get_converter(self):
        """
        Create Docling DocumentConverter only when needed.

        This keeps TXT / Markdown ingestion lightweight.
        """

        if self._converter is None:

            try:

                from docling.document_converter import (
                    DocumentConverter,
                )

            except ImportError as exc:

                raise IngestionError(
                    "Docling is required for "
                    "PDF/DOCX/PPTX/HTML/Image ingestion."
                ) from exc

            self._converter = (
                DocumentConverter()
            )

        return self._converter


    # ========================================================
    # Main public API
    # ========================================================
    #这个模块的唯一核心入口
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

        This method is deliberately kept as the stable
        single-file atomic operation.

        Future BatchIngestion will call this method repeatedly.
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
        # 3. Detect file format
        # ----------------------------------------------------
        #判断 PDF / DOCX / TXT / HTML……
        source_format = (
            detect_source_format(
                file_path
            )
        )


        # ----------------------------------------------------
        # 4. Build stable document identity
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
        # 5. Parse content
        # ----------------------------------------------------

        parser_name = None

        page_count = None

        docling_json_path = None


        # ====================================================
        # Lightweight text path
        # ====================================================
        #表示简单格式走轻量路线。
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
        # Docling path
        # ====================================================
        #否则，复杂格式交给 Docling
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
                # Export a readable representation.
                #
                # Markdown keeps headings / lists / tables
                # better than plain text for later chunking.
                # --------------------------------------------

                normalized_text = (
                    docling_document
                    .export_to_markdown()
                )

                parser_name = "docling"


                # --------------------------------------------
                # Page count if available.
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
                # Preserve Docling's richer structural JSON.
                #
                # Later Structure-aware Chunking can use it.
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
                    #保存 Docling 的完整结构，留给下一阶段 Structure-aware Chunking
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

            raise IngestionError(
                "No usable text was extracted "
                f"from {file_path.name}."
            )


        # ----------------------------------------------------
        # 7. Merge technical + custom metadata
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
                str(docling_json_path)
                if docling_json_path
                else None
            ),

            "text_representation": (
                "markdown"
                if parser_name == "docling"
                else "plain_text"
            ),
        }

        #给未来 Manifest / Batch Ingestion 留扩展口
        if extra_metadata:

            metadata.update(
                extra_metadata
            )


        # ----------------------------------------------------
        # 8. Build project's unified KnowledgeDocument
        # ----------------------------------------------------
        #最重要的一步，把外部解析器结果正式转换为我们项目内部的数据合同
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
        # 9. Persist project's unified representation
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
        # 10. Return unified document
        # ----------------------------------------------------

        return document