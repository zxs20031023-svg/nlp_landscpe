from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

from .config import PATHS, load_json


def resolve_loader(file_path: Path):
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        return PyPDFLoader(str(file_path))
    if suffix in {".docx", ".doc"}:
        from langchain_community.document_loaders import Docx2txtLoader

        return Docx2txtLoader(str(file_path))
    from langchain_community.document_loaders import TextLoader

    return TextLoader(str(file_path), encoding="utf-8")


def _filter_empty_documents(documents: Iterable) -> list[Document]:
    filtered: list[Document] = []
    for doc in documents:
        content = getattr(doc, "page_content", "")
        if content and content.strip():
            filtered.append(doc)
    return filtered


def _extract_pdf_with_pymupdf(file_path: Path) -> list[Document]:
    import pymupdf

    pdf = pymupdf.open(str(file_path))
    docs: list[Document] = []
    for page_index, page in enumerate(pdf):
        text = page.get_text("text").strip()
        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": file_path.name, "page": page_index},
                )
            )
    return docs


def _extract_pdf_with_ocr(file_path: Path) -> list[Document]:
    import pymupdf
    from rapidocr_onnxruntime import RapidOCR

    pdf = pymupdf.open(str(file_path))
    ocr = RapidOCR()
    docs: list[Document] = []

    for page_index, page in enumerate(pdf):
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
        result, _ = ocr(pix.tobytes("png"))
        if not result:
            continue

        page_text = "\n".join(item[1] for item in result if len(item) > 1 and item[1].strip()).strip()
        if page_text:
            docs.append(
                Document(
                    page_content=page_text,
                    metadata={"source": file_path.name, "page": page_index, "loader": "rapidocr"},
                )
            )
    return docs


def _load_curated_pdf_fallback(file_path: Path) -> list[Document]:
    if not PATHS.knowledge_aliases_file.exists():
        return []

    aliases = load_json(PATHS.knowledge_aliases_file)
    alias_name = aliases.get(file_path.name.lower())
    if not alias_name:
        return []

    alias_path = PATHS.knowledge_base_dir / alias_name
    if not alias_path.exists():
        return []

    loader = resolve_loader(alias_path)
    return _filter_empty_documents(loader.load())


def _pdf_page_count(file_path: Path) -> int:
    import pymupdf

    pdf = pymupdf.open(str(file_path))
    return len(pdf)


def load_documents(file_path: Path):
    suffix = file_path.suffix.lower()
    loader = resolve_loader(file_path)
    documents = _filter_empty_documents(loader.load())

    if suffix == ".pdf" and not documents:
        documents = _load_curated_pdf_fallback(file_path)

    if suffix == ".pdf" and not documents:
        documents = _extract_pdf_with_pymupdf(file_path)

    if suffix == ".pdf" and not documents:
        if _pdf_page_count(file_path) <= 12:
            documents = _extract_pdf_with_ocr(file_path)
        else:
            raise ValueError(
                f"PDF 无法直接提取文本，且页数较多不适合在线 OCR：{file_path.name}。"
                "请优先上传可复制文本的 TXT/DOCX，或为该文件提供预处理文本资源。"
            )

    if not documents:
        raise ValueError(f"文件解析结果为空：{file_path.name}")
    return documents


def split_documents(documents: Iterable, chunk_size: int = 500, chunk_overlap: int = 80):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    normalized_documents = _filter_empty_documents(documents)
    if not normalized_documents:
        raise ValueError("文档切分结果为空")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " "],
    )
    splits = splitter.split_documents(normalized_documents)
    if not splits:
        raise ValueError("文档切分结果为空")
    return splits


def write_uploaded_file(uploaded_file, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / uploaded_file.name
    file_path.write_bytes(uploaded_file.getbuffer())
    return file_path


def extract_uploaded_text(uploaded_file, temp_dir: Path) -> str:
    file_path = write_uploaded_file(uploaded_file, temp_dir)
    try:
        docs = load_documents(file_path)
        return "\n".join(doc.page_content for doc in docs if doc.page_content)
    finally:
        file_path.unlink(missing_ok=True)


def temporary_directory() -> tempfile.TemporaryDirectory[str]:
    return tempfile.TemporaryDirectory(prefix="landscape_nlp_")
