from __future__ import annotations

import argparse
import hashlib
import json
import math
import re

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import BaseModel

from app.rag.manifest import load_manifest


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "knowledge"
    / "manifests"
    / "knowledge_sources.jsonl"
)


DEFAULT_DOCUMENT_DIR = (
    PROJECT_ROOT
    / "knowledge"
    / "processed"
    / "documents"
)


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "knowledge"
    / "processed"
    / "chunks"
)


DEFAULT_CHUNKS_PATH = (
    DEFAULT_OUTPUT_DIR
    / "knowledge_chunks.jsonl"
)


DEFAULT_REPORT_PATH = (
    DEFAULT_OUTPUT_DIR
    / "chunking_report.json"
)


# ============================================================
# 2. Errors
# ============================================================

class ChunkingError(RuntimeError):
    """
    Raised when a KnowledgeDocument cannot be chunked safely.
    """
    pass


# ============================================================
# 3. Unified KnowledgeChunk
# ============================================================

class KnowledgeChunk(BaseModel):
    """
    Chunk 之后的统一数据合同。

    后面的：

        Embedding
        Qdrant
        Retriever

    都只应该依赖 KnowledgeChunk，
    不需要再知道底层来自 PDF / HTML / PPTX。
    """

    # ---------- identity ----------

    chunk_id: str

    # processed KnowledgeDocument ID
    document_id: str

    # Source Scanner / Manifest ID
    manifest_document_id: str

    chunk_index: int


    # ---------- chunk content ----------

    text: str

    title: str

    section: Optional[str] = None

    # PDF 当前可以精确追踪页码。
    # HTML/PPTX 第一版允许 None。
    page: Optional[int] = None


    # ---------- source ----------

    source_name: str

    source_format: str

    language: str


    # ---------- industrial metadata ----------

    equipment_type: Optional[str] = None

    equipment_model: Optional[str] = None

    fault_type: Optional[str] = None


    # ---------- provenance ----------

    source: Optional[str] = None

    source_url: Optional[str] = None

    document_type: Optional[str] = None

    knowledge_scope: Optional[str] = None

    authority_level: Optional[str] = None

    relative_path: Optional[str] = None


    # ---------- chunk technical metadata ----------

    chunking_strategy: str

    char_count: int

    estimated_tokens: int


# ============================================================
# 4. Chunking config
# ============================================================

@dataclass(frozen=True)
class ChunkingConfig:

    # 当前是近似 Token，不绑定某一个 Embedding tokenizer。
    max_tokens: int = 450

    # 相邻 Chunk 保留少量上下文。
    overlap_tokens: int = 60

    # 避免产生大量过小 Chunk。
    min_tokens: int = 40


# ============================================================
# 5. Regex / cleanup rules
# ============================================================

MARKDOWN_HEADING_RE = re.compile(
    r"^#{1,6}\s+(?P<title>.+?)\s*$"
)


# 支持：
#
# 四、案例分析
# 第一章 xxx
# 1. xxx
# 2.1 xxx
#
CHINESE_HEADING_RE = re.compile(
    r"^(?:"
    r"第[一二三四五六七八九十百零0-9]+[章节部分]"
    r"|"
    r"[一二三四五六七八九十]+[、.]"
    r"|"
    r"\d+(?:\.\d+)*[、.\s]"
    r")"
    r"\s*(?P<title>.+)$"
)


# 网页里常见：
#
# - [Pumps](https://...)
#
LINK_ONLY_BULLET_RE = re.compile(
    r"^\s*[-*+]\s*"
    r"\[[^\]]+\]"
    r"\([^\)]+\)"
    r"\s*$"
)


PAGE_MARKER_RE = re.compile(
    r"<!--\s*page:\s*\d+\s*-->",
    re.IGNORECASE,
)


IMAGE_PLACEHOLDER_RE = re.compile(
    r"<!--\s*image\s*-->",
    re.IGNORECASE,
)


CJK_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]"
)


BOILERPLATE_LINES = {
    "categories",
    "category",
    "tags",
    "tag",
}


# ============================================================
# 6. Approximate token counting
# ============================================================

def estimate_tokens(
    text: str,
) -> int:
    """
    第一版不绑定具体 Embedding 模型 tokenizer。

    粗略估计：

        中文字符 ≈ 1 token
        非中文 ≈ 4 chars / token

    这里的目的不是计算 API 账单，
    而是控制 Chunk 大小。

    真正选定 Embedding 模型后，
    第二遍可以换成对应 tokenizer。
    """

    if not text:
        return 0

    cjk_count = len(
        CJK_RE.findall(text)
    )

    non_cjk = (
        CJK_RE.sub(
            "",
            text,
        )
    )

    non_cjk_nonspace = len(
        re.sub(
            r"\s+",
            "",
            non_cjk,
        )
    )

    return max(
        1,

        cjk_count
        + math.ceil(
            non_cjk_nonspace / 4
        ),
    )


# ============================================================
# 7. Plain text normalization
# ============================================================

def normalize_plain_text(
    text: str,
) -> str:
    """
    处理 PDF 等来源中的硬换行。

    原始：

        high-pressure pumps are key compo
        nents in oil and gas systems

    第一版会至少整理成：

        high-pressure pumps are key compo nents ...

    注意：
    当前不会激进地猜测所有断词并自动拼接，
    避免错误修改真实技术术语。
    """

    text = (
        text
        .replace(
            "\r\n",
            "\n",
        )
        .replace(
            "\r",
            "\n",
        )
    )


    # 如果 KnowledgeDocument 中存在 page marker，
    # Chunking 时去掉，因为 page 已进入独立 Metadata。
    text = PAGE_MARKER_RE.sub(
        "",
        text,
    )


    paragraphs: list[str] = []


    # 空行 = 段落边界
    for raw_paragraph in re.split(
        r"\n\s*\n+",
        text,
    ):

        lines = [

            re.sub(
                r"\s+",
                " ",
                line,
            ).strip()

            for line
            in raw_paragraph.split("\n")

            if line.strip()
        ]


        if not lines:
            continue


        # 段落内部的视觉换行合并
        paragraph = (
            " ".join(lines)
            .strip()
        )


        paragraph = re.sub(
            r"\s{2,}",
            " ",
            paragraph,
        )


        if paragraph:
            paragraphs.append(
                paragraph
            )


    return "\n\n".join(
        paragraphs
    )


# ============================================================
# 8. Markdown cleanup
# ============================================================

def clean_markdown(
    text: str,
) -> str:
    """
    低风险清理 HTML / PPTX Docling Markdown。

    当前只处理已经被真实数据验证的噪声：

        Categories
        Tags
        link-only列表
        <!-- image -->

    不进行激进网页正文抽取。
    """

    text = (
        text
        .replace(
            "\r\n",
            "\n",
        )
        .replace(
            "\r",
            "\n",
        )
    )


    # PPTX 当前没有做多模态，
    # 所以纯图片占位符不作为知识 Chunk。
    text = IMAGE_PLACEHOLDER_RE.sub(
        "",
        text,
    )


    cleaned_lines: list[str] = []


    for raw_line in text.split("\n"):

        line = raw_line.strip()


        if not line:

            cleaned_lines.append("")

            continue


        # HTML实际出现过的Categories / Tags
        if (
            line.lower()
            in BOILERPLATE_LINES
        ):

            continue


        # 删除纯网页分类链接列表
        if LINK_ONLY_BULLET_RE.match(
            line
        ):

            continue


        cleaned_lines.append(
            line
        )


    cleaned = "\n".join(
        cleaned_lines
    )


    cleaned = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned,
    )


    return cleaned.strip()


# ============================================================
# 9. Heading recognition
# ============================================================

def detect_heading(
    line: str,
) -> Optional[str]:

    line = line.strip()


    # Markdown:
    #
    # ## Working on Mud Pumps
    #
    markdown_match = (
        MARKDOWN_HEADING_RE.match(
            line
        )
    )


    if markdown_match:

        return (
            markdown_match
            .group("title")
            .strip()
        )


    # Chinese:
    #
    # 四、案例分析
    #
    chinese_match = (
        CHINESE_HEADING_RE.match(
            line
        )
    )


    if chinese_match:

        title = (
            chinese_match
            .group("title")
            .strip()
        )


        # 避免把很长正文误识别成标题
        if 1 <= len(title) <= 80:

            return line


    return None


# ============================================================
# 10. Markdown → sections
# ============================================================

def markdown_sections(
    text: str,
    default_section: Optional[str],
) -> list[
    tuple[
        Optional[str],
        str,
    ]
]:
    """
    HTML / PPTX 第一版 Structure-aware 的核心。

    Heading 出现时切换 section，
    但正文仍然在 section 内继续按段落和 Token 合并。
    """

    text = clean_markdown(
        text
    )


    current_section = (
        default_section
    )


    body_lines: list[str] = []


    sections: list[
        tuple[
            Optional[str],
            str,
        ]
    ] = []


    def flush() -> None:

        nonlocal body_lines


        body = normalize_plain_text(
            "\n".join(
                body_lines
            )
        )


        if body:

            sections.append(
                (
                    current_section,
                    body,
                )
            )


        body_lines = []


    for line in text.split("\n"):

        heading = detect_heading(
            line
        )


        if heading is not None:

            flush()

            current_section = (
                heading
            )

            continue


        body_lines.append(
            line
        )


    flush()


    return sections


# ============================================================
# 11. Sentence splitting
# ============================================================

def split_sentences(
    text: str,
) -> list[str]:
    """
    当一个段落过长时，优先按句子继续拆。

    不直接按固定字符粗暴切断。
    """

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


    if not text:
        return []


    parts = re.split(

        r"(?<=[。！？!?])"
        r"|"
        r"(?<=[.!?])\s+",

        text,
    )


    return [

        part.strip()

        for part in parts

        if part.strip()
    ]


# ============================================================
# 12. Last-resort hard split
# ============================================================

def hard_split(
    text: str,
    max_tokens: int,
) -> list[str]:
    """
    只有“单个句子本身就特别长”时才启用。

    这是兜底，不是主策略。
    """

    if (
        estimate_tokens(text)
        <= max_tokens
    ):

        return [
            text.strip()
        ]


    cjk_count = len(
        CJK_RE.findall(text)
    )


    cjk_ratio = (
        cjk_count
        / max(
            1,
            len(text),
        )
    )


    # 中文：
    # max_tokens ≈ max_chars
    #
    # 英文：
    # 约 4 chars / token
    #
    max_chars = (

        max_tokens

        if cjk_ratio >= 0.35

        else
        max_tokens * 4
    )


    max_chars = max(
        200,
        int(max_chars),
    )


    return [

        text[
            index:
            index + max_chars
        ].strip()

        for index
        in range(
            0,
            len(text),
            max_chars,
        )

        if text[
            index:
            index + max_chars
        ].strip()
    ]


# ============================================================
# 13. Text → semantic blocks
# ============================================================

def text_blocks(
    text: str,
    max_tokens: int,
) -> list[str]:
    """
    优先级：

        Paragraph
            ↓
        Sentence
            ↓
        Hard split

    而不是一开始就固定长度切割。
    """

    normalized = (
        normalize_plain_text(
            text
        )
    )


    paragraphs = [

        paragraph.strip()

        for paragraph
        in re.split(
            r"\n\s*\n+",
            normalized,
        )

        if paragraph.strip()
    ]


    blocks: list[str] = []


    for paragraph in paragraphs:

        if (
            estimate_tokens(
                paragraph
            )
            <= max_tokens
        ):

            blocks.append(
                paragraph
            )

            continue


        for sentence in split_sentences(
            paragraph
        ):

            blocks.extend(

                hard_split(
                    sentence,
                    max_tokens,
                )
            )


    return blocks


# ============================================================
# 14. Overlap
# ============================================================

def trailing_overlap(
    blocks: list[str],
    overlap_tokens: int,
) -> list[str]:
    """
    从上一 Chunk 尾部保留少量上下文。

    overlap不是越大越好。
    当前只保留约60 token。
    """

    if overlap_tokens <= 0:

        return []


    selected: list[str] = []

    total = 0


    for block in reversed(
        blocks
    ):

        block_tokens = (
            estimate_tokens(
                block
            )
        )


        if (
            selected
            and total + block_tokens
            > overlap_tokens
        ):

            break


        # 单个Block本身已经超过overlap，
        # 尝试只保留其末尾几个句子。
        if (
            not selected
            and block_tokens
            > overlap_tokens
        ):

            sentences = (
                split_sentences(
                    block
                )
            )


            if sentences:

                tail: list[str] = []

                tail_total = 0


                for sentence in reversed(
                    sentences
                ):

                    sentence_tokens = (
                        estimate_tokens(
                            sentence
                        )
                    )


                    if (
                        tail
                        and tail_total
                        + sentence_tokens
                        > overlap_tokens
                    ):

                        break


                    tail.insert(
                        0,
                        sentence,
                    )


                    tail_total += (
                        sentence_tokens
                    )


                return tail


            break


        selected.insert(
            0,
            block,
        )


        total += (
            block_tokens
        )


    return selected


# ============================================================
# 15. Blocks → chunks
# ============================================================

def pack_blocks(
    blocks: list[str],
    config: ChunkingConfig,
) -> list[str]:
    """
    将Paragraph / Sentence blocks合并到目标大小。

    尽量保持语义边界。
    """

    chunks: list[str] = []

    current: list[str] = []

    current_tokens = 0


    for block in blocks:

        block_tokens = (
            estimate_tokens(
                block
            )
        )


        # 当前Chunk再加入block会超限
        if (
            current
            and current_tokens
            + block_tokens
            > config.max_tokens
        ):

            chunks.append(

                "\n\n"
                .join(current)
                .strip()
            )


            # 新Chunk继承少量上一Chunk尾部上下文
            current = trailing_overlap(

                current,

                config.overlap_tokens,
            )


            current_tokens = sum(

                estimate_tokens(
                    item
                )

                for item
                in current
            )


        current.append(
            block
        )


        current_tokens += (
            block_tokens
        )


    # 最后一块
    if current:

        tail = (

            "\n\n"
            .join(current)
            .strip()
        )


        # 如果尾块特别短，
        # 并且并回前一个不会明显超限，
        # 则减少碎片化。
        if (
            chunks

            and estimate_tokens(
                tail
            )
            < config.min_tokens

            and estimate_tokens(
                chunks[-1]
                + "\n\n"
                + tail
            )
            <= int(
                config.max_tokens
                * 1.15
            )
        ):

            chunks[-1] = (

                chunks[-1]
                + "\n\n"
                + tail
            ).strip()

        else:

            chunks.append(
                tail
            )


    return [

        chunk

        for chunk in chunks

        if chunk.strip()
    ]


# ============================================================
# 16. Stable chunk ID
# ============================================================

def stable_chunk_id(
    *,
    document_id: str,
    chunk_index: int,
    page: Optional[int],
    section: Optional[str],
    text: str,
) -> str:
    """
    Chunk内容变化时，ID也变化。

    以后用于：
        Qdrant upsert
        chunk version tracking
    """

    payload = "|".join(
        [
            document_id,

            str(
                chunk_index
            ),

            str(
                page or ""
            ),

            section or "",

            text,
        ]
    )


    digest = hashlib.sha256(

        payload.encode(
            "utf-8"
        )

    ).hexdigest()


    return (
        f"chk_{digest[:20]}"
    )


# ============================================================
# 17. Build KnowledgeChunk
# ============================================================

def make_chunk(
    *,
    document: dict[str, Any],
    text: str,
    chunk_index: int,
    strategy: str,
    section: Optional[str] = None,
    page: Optional[int] = None,
) -> KnowledgeChunk:

    metadata = (
        document.get(
            "metadata"
        )
        or {}
    )


    return KnowledgeChunk(

        chunk_id=stable_chunk_id(

            document_id=(
                document[
                    "document_id"
                ]
            ),

            chunk_index=(
                chunk_index
            ),

            page=page,

            section=section,

            text=text,
        ),


        document_id=(
            document[
                "document_id"
            ]
        ),


        manifest_document_id=str(

            metadata.get(
                "manifest_document_id"
            )
            or ""
        ),


        chunk_index=(
            chunk_index
        ),


        text=text,


        title=str(

            document.get(
                "title"
            )

            or document.get(
                "source_name"
            )

            or "untitled"
        ),


        section=section,

        page=page,


        source_name=str(

            document.get(
                "source_name"
            )
            or ""
        ),


        source_format=str(

            document.get(
                "source_format"
            )
            or "unknown"
        ),


        language=str(

            document.get(
                "language"
            )
            or "unknown"
        ),


        equipment_type=(
            document.get(
                "equipment_type"
            )
        ),


        equipment_model=(
            document.get(
                "equipment_model"
            )
        ),


        fault_type=(
            metadata.get(
                "fault_type"
            )
        ),


        source=(
            metadata.get(
                "source"
            )
        ),


        source_url=(
            metadata.get(
                "source_url"
            )
        ),


        document_type=(
            metadata.get(
                "document_type"
            )
        ),


        knowledge_scope=(
            metadata.get(
                "knowledge_scope"
            )
        ),


        authority_level=(
            metadata.get(
                "authority_label"
            )
        ),


        relative_path=(
            metadata.get(
                "relative_path"
            )
        ),


        chunking_strategy=(
            strategy
        ),


        char_count=len(
            text
        ),


        estimated_tokens=(
            estimate_tokens(
                text
            )
        ),
    )


# ============================================================
# 18. JSON loader
# ============================================================

def load_json(
    path: Path,
) -> dict[str, Any]:

    try:

        return json.loads(

            path.read_text(
                encoding="utf-8"
            )
        )

    except Exception as exc:

        raise ChunkingError(
            "Unable to read JSON: "
            f"{path}: {exc}"
        ) from exc


# ============================================================
# 19. Select current active KnowledgeDocuments
# ============================================================

def load_active_documents(
    *,
    manifest_path: Path,
    document_dir: Path,
) -> list[
    dict[str, Any]
]:
    """
    只处理：

        review_status = approved
        parse_status  = success

    的当前正式知识源。

    这样可以避免之前测试失败留下的旧artifact
    被误加入正式Chunk库。
    """

    entries = load_manifest(
        manifest_path
    )


    active_manifest_ids = {

        entry.document_id

        for entry in entries

        if (
            entry.review_status
            == "approved"

            and entry.parse_status
            == "success"
        )
    }


    documents: list[
        dict[str, Any]
    ] = []


    for path in sorted(
        document_dir.glob(
            "*.knowledge.json"
        )
    ):

        document = load_json(
            path
        )


        metadata = (
            document.get(
                "metadata"
            )
            or {}
        )


        manifest_document_id = (
            metadata.get(
                "manifest_document_id"
            )
        )


        if (
            manifest_document_id
            in active_manifest_ids
        ):

            documents.append(
                document
            )


    # 防止Manifest显示success，
    # 但processed artifact丢失。
    found_manifest_ids = {

        (
            document
            .get(
                "metadata",
                {}
            )
            .get(
                "manifest_document_id"
            )
        )

        for document
        in documents
    }


    missing = (

        active_manifest_ids

        - found_manifest_ids
    )


    if missing:

        raise ChunkingError(

            "Some approved + successfully "
            "ingested Manifest documents "
            "have no KnowledgeDocument artifact: "

            + ", ".join(
                sorted(missing)
            )
        )


    return documents


# ============================================================
# 20. Resolve PDF pages artifact
# ============================================================

def resolve_pdf_pages_path(
    document: dict[str, Any],
) -> Path:

    metadata = (
        document.get(
            "metadata"
        )
        or {}
    )


    raw_path = (
        metadata.get(
            "pdf_pages_path"
        )
    )


    if not raw_path:

        raise ChunkingError(

            "PDF "
            f"{document.get('source_name')} "
            "has no pdf_pages_path metadata."
        )


    path = Path(
        raw_path
    )


    if not path.is_absolute():

        path = (
            PROJECT_ROOT
            / path
        )


    path = path.resolve()


    if not path.exists():

        raise ChunkingError(
            "PDF page artifact "
            f"not found: {path}"
        )


    return path


# ============================================================
# 21. PDF page-aware chunking
# ============================================================

def chunk_pdf(
    document: dict[str, Any],
    config: ChunkingConfig,
) -> list[KnowledgeChunk]:
    """
    PDF第一版策略：

        page
        → paragraph
        → sentence
        → pack

    Chunk不跨页。

    好处：
        citation可以准确保留page。
    """

    pages_path = (
        resolve_pdf_pages_path(
            document
        )
    )


    page_artifact = load_json(
        pages_path
    )


    chunks: list[
        KnowledgeChunk
    ] = []


    chunk_index = 0


    for page in (
        page_artifact
        .get(
            "pages",
            [],
        )
    ):

        page_number = int(
            page.get(
                "page_number"
            )
        )


        page_text = (
            normalize_plain_text(

                str(
                    page.get(
                        "text"
                    )
                    or ""
                )
            )
        )


        if not page_text:

            continue


        blocks = text_blocks(

            page_text,

            config.max_tokens,
        )


        for chunk_text in pack_blocks(

            blocks,

            config,
        ):

            chunks.append(

                make_chunk(

                    document=document,

                    text=chunk_text,

                    chunk_index=(
                        chunk_index
                    ),

                    strategy=(
                        "page_aware_paragraph"
                    ),

                    page=(
                        page_number
                    ),

                    section=None,
                )
            )


            chunk_index += 1


    return chunks


# ============================================================
# 22. HTML / PPTX / Markdown chunking
# ============================================================

def chunk_markdown_document(
    document: dict[str, Any],
    config: ChunkingConfig,
) -> list[KnowledgeChunk]:
    """
    Docling已经把HTML/PPTX等转换成Markdown。

    当前利用：

        Heading
        Paragraph
        Token size

    做Section-aware Chunking。
    """

    raw_text = str(
        document.get(
            "text"
        )
        or ""
    )


    title = (

        str(
            document.get(
                "title"
            )
            or ""
        )

        or None
    )


    sections = markdown_sections(

        raw_text,

        default_section=title,
    )


    chunks: list[
        KnowledgeChunk
    ] = []


    chunk_index = 0


    for (
        section,
        section_text,
    ) in sections:

        blocks = text_blocks(

            section_text,

            config.max_tokens,
        )


        for chunk_text in pack_blocks(

            blocks,

            config,
        ):

            chunks.append(

                make_chunk(

                    document=document,

                    text=chunk_text,

                    chunk_index=(
                        chunk_index
                    ),

                    strategy=(
                        "markdown_section"
                    ),

                    page=None,

                    section=section,
                )
            )


            chunk_index += 1


    return chunks


# ============================================================
# 23. Route document to strategy
# ============================================================

def chunk_document(
    document: dict[str, Any],
    config: ChunkingConfig,
) -> list[KnowledgeChunk]:

    source_format = str(

        document.get(
            "source_format"
        )
        or ""

    ).lower()


    if source_format == "pdf":

        return chunk_pdf(
            document,
            config,
        )


    # HTML / PPTX / DOCX / TXT / Markdown
    #
    # 第一版统一利用文本结构。
    return chunk_markdown_document(
        document,
        config,
    )


# ============================================================
# 24. Save chunks
# ============================================================

def save_chunks(
    chunks: Iterable[
        KnowledgeChunk
    ],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(

        parents=True,

        exist_ok=True,
    )


    with output_path.open(

        "w",

        encoding="utf-8",

    ) as file:

        for chunk in chunks:

            file.write(

                chunk.model_dump_json()
                + "\n"
            )


# ============================================================
# 25. Main Chunking pipeline
# ============================================================

def run_chunking(
    *,
    manifest_path: Path,
    document_dir: Path,
    output_path: Path,
    report_path: Path,
    config: ChunkingConfig,
) -> dict[str, Any]:

    documents = (
        load_active_documents(

            manifest_path=(
                manifest_path
            ),

            document_dir=(
                document_dir
            ),
        )
    )


    all_chunks: list[
        KnowledgeChunk
    ] = []


    per_document: list[
        dict[str, Any]
    ] = []


    for document in documents:

        document_chunks = (
            chunk_document(

                document,

                config,
            )
        )


        if not document_chunks:

            raise ChunkingError(

                "No chunks produced for "

                f"{document.get('source_name')}"
            )


        all_chunks.extend(
            document_chunks
        )


        token_sizes = [

            chunk.estimated_tokens

            for chunk
            in document_chunks
        ]


        item = {

            "document_id":
            document.get(
                "document_id"
            ),

            "source_name":
            document.get(
                "source_name"
            ),

            "source_format":
            document.get(
                "source_format"
            ),

            "chunks":
            len(
                document_chunks
            ),

            "min_estimated_tokens":
            min(
                token_sizes
            ),

            "max_estimated_tokens":
            max(
                token_sizes
            ),

            "avg_estimated_tokens":
            round(
                sum(token_sizes)
                / len(token_sizes),
                2,
            ),
        }


        per_document.append(
            item
        )


        print(

            "[chunk] "

            f"{item['source_name']} | "

            f"format="
            f"{item['source_format']} | "

            f"chunks="
            f"{item['chunks']} | "

            f"avg_tokens="
            f"{item['avg_estimated_tokens']}"
        )


    save_chunks(
        all_chunks,
        output_path,
    )


    report = {

        "documents":
        len(documents),

        "chunks":
        len(all_chunks),

        "max_tokens":
        config.max_tokens,

        "overlap_tokens":
        config.overlap_tokens,

        "min_tokens":
        config.min_tokens,

        "output_path":
        str(
            output_path.resolve()
        ),

        "per_document":
        per_document,
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


# ============================================================
# 26. CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(

        description=(
            "Structure-aware / page-aware "
            "industrial knowledge chunking"
        )
    )


    parser.add_argument(

        "--manifest",

        type=Path,

        default=(
            DEFAULT_MANIFEST_PATH
        ),
    )


    parser.add_argument(

        "--document-dir",

        type=Path,

        default=(
            DEFAULT_DOCUMENT_DIR
        ),
    )


    parser.add_argument(

        "--output",

        type=Path,

        default=(
            DEFAULT_CHUNKS_PATH
        ),
    )


    parser.add_argument(

        "--report",

        type=Path,

        default=(
            DEFAULT_REPORT_PATH
        ),
    )


    parser.add_argument(

        "--max-tokens",

        type=int,

        default=450,
    )


    parser.add_argument(

        "--overlap-tokens",

        type=int,

        default=60,
    )


    parser.add_argument(

        "--min-tokens",

        type=int,

        default=40,
    )


    args = parser.parse_args()


    # ---------- config validation ----------

    if args.max_tokens <= 0:

        raise SystemExit(
            "--max-tokens must be > 0"
        )


    if args.overlap_tokens < 0:

        raise SystemExit(
            "--overlap-tokens must be >= 0"
        )


    if (
        args.overlap_tokens
        >= args.max_tokens
    ):

        raise SystemExit(
            "--overlap-tokens must be "
            "smaller than --max-tokens"
        )


    if args.min_tokens < 0:

        raise SystemExit(
            "--min-tokens must be >= 0"
        )


    config = ChunkingConfig(

        max_tokens=(
            args.max_tokens
        ),

        overlap_tokens=(
            args.overlap_tokens
        ),

        min_tokens=(
            args.min_tokens
        ),
    )


    report = run_chunking(

        manifest_path=(
            args.manifest
            .resolve()
        ),

        document_dir=(
            args.document_dir
            .resolve()
        ),

        output_path=(
            args.output
            .resolve()
        ),

        report_path=(
            args.report
            .resolve()
        ),

        config=config,
    )


    print()

    print(
        "[chunk] summary"
    )


    print(

        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":

    main()