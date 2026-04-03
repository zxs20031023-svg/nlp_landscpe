from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path

from .config import PATHS
from .loaders import load_documents, split_documents, write_uploaded_file
from .models import IngestionResult, KnowledgeBaseStats, KnowledgeFileRecord


class KnowledgeBaseManager:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        embedding_model: str,
        persist_directory: Path,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.manifest_path = persist_directory / "manifest.json"

    def _load_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {"documents": []}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _save_manifest(self, manifest: dict) -> None:
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _build_embeddings(self):
        from langchain_openai import OpenAIEmbeddings

        # Some OpenAI-compatible embedding providers reject batches larger than 10.
        return OpenAIEmbeddings(
            api_key=self.api_key,
            base_url=self.base_url or None,
            model=self.embedding_model,
            chunk_size=10,
            check_embedding_ctx_length=False,
        )

    @staticmethod
    def _file_hash(file_path: Path) -> str:
        return hashlib.sha256(file_path.read_bytes()).hexdigest()

    def get_stats(self) -> KnowledgeBaseStats:
        manifest = self._load_manifest()
        records = [KnowledgeFileRecord(**item) for item in manifest.get("documents", [])]
        ready = self.persist_directory.exists() and bool(
            list(self.persist_directory.glob("*.sqlite3")) or records
        )
        return KnowledgeBaseStats(ready=ready, document_count=len(records), documents=records)

    def ingest_uploaded_file(self, uploaded_file) -> IngestionResult:
        staging_dir = self.persist_directory / "_staging"
        file_path = write_uploaded_file(uploaded_file, staging_dir)
        try:
            return self.ingest_file(file_path=file_path, original_name=uploaded_file.name)
        finally:
            file_path.unlink(missing_ok=True)

    def ingest_file(self, file_path: Path, original_name: str | None = None) -> IngestionResult:
        if not self.api_key:
            raise ValueError("构建知识库前需要先填写 API Key。")

        original_name = original_name or file_path.name
        file_hash = self._file_hash(file_path)
        manifest = self._load_manifest()
        known_hashes = {item["file_hash"] for item in manifest.get("documents", [])}
        if file_hash in known_hashes:
            return IngestionResult(
                filename=original_name,
                chunk_count=0,
                file_hash=file_hash,
                skipped=True,
            )

        docs = load_documents(file_path)
        splits = split_documents(docs)
        for index, doc in enumerate(splits):
            doc.metadata.update(
                {
                    "source": original_name,
                    "chunk_index": index,
                    "file_hash": file_hash,
                }
            )

        from langchain_community.vectorstores import Chroma

        vectorstore = Chroma(
            persist_directory=str(self.persist_directory),
            embedding_function=self._build_embeddings(),
        )
        vectorstore.add_documents(
            documents=splits,
            ids=[f"{file_hash}-{index}-{uuid.uuid4().hex[:8]}" for index in range(len(splits))],
        )

        manifest.setdefault("documents", []).append(
            {
                "filename": original_name,
                "file_hash": file_hash,
                "chunk_count": len(splits),
                "ingested_at": datetime.now().isoformat(timespec="seconds"),
                "source_path": str(file_path),
            }
        )
        self._save_manifest(manifest)
        return IngestionResult(filename=original_name, chunk_count=len(splits), file_hash=file_hash)

    def list_preferred_bundled_files(self) -> list[Path]:
        preferred_suffixes = {".txt", ".docx", ".doc"}
        files = [
            path for path in sorted(PATHS.knowledge_base_dir.glob("*"))
            if path.is_file() and path.suffix.lower() in preferred_suffixes
        ]
        return files

    def ingest_bundled_resources(self) -> list[IngestionResult]:
        results: list[IngestionResult] = []
        for file_path in self.list_preferred_bundled_files():
            results.append(self.ingest_file(file_path=file_path, original_name=file_path.name))
        return results

    def retrieve(self, query: str, top_k: int = 4) -> list:
        if not self.persist_directory.exists():
            return []

        from langchain_community.vectorstores import Chroma

        vectorstore = Chroma(
            persist_directory=str(self.persist_directory),
            embedding_function=self._build_embeddings(),
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        return retriever.invoke(query)
