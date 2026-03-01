from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def _normalize_gemini_embedding_model(model_name: str) -> str:
    name = (model_name or "").strip() or "models/embedding-001"
    lowered = name.lower()
    # Gemini embedding endpoint expects the embedding-001 model family.
    if "text-embedding" in lowered:
        return "models/embedding-001"
    if lowered in {"embedding-004", "models/embedding-004"}:
        return "models/embedding-001"
    if not name.startswith("models/"):
        return f"models/{name}"
    return name


def _resolve_embedding_provider(config, forced_provider: Optional[str] = None) -> str:
    raw = (forced_provider or config.get("EMBEDDING_PROVIDER") or "auto").strip().lower()
    if raw not in {"auto", "gemini", "local"}:
        raw = "auto"
    if raw == "auto":
        return "gemini" if (config.get("GEMINI_API_KEY") or "").strip() else "local"
    return raw


def _get_local_embeddings(config):
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except Exception as exc:
        raise RuntimeError(
            "Local embeddings require sentence-transformers. Install with: pip install sentence-transformers"
        ) from exc

    model_name = (config.get("LOCAL_EMBEDDING_MODEL") or "").strip() or "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


def _get_embeddings(config):
    provider = _resolve_embedding_provider(config)
    if provider == "local":
        return _get_local_embeddings(config), "local"

    gemini_key = (config.get("GEMINI_API_KEY") or "").strip()
    if not gemini_key:
        raise RuntimeError("GEMINI_API_KEY is not configured. Set GEMINI_API_KEY or use EMBEDDING_PROVIDER=local.")

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    model_name = _normalize_gemini_embedding_model(config.get("EMBEDDING_MODEL", "models/embedding-001"))
    return (
        GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=gemini_key,
            task_type="retrieval_document",
        ),
        "gemini",
    )


def _get_query_embeddings(config, forced_provider: Optional[str] = None):
    provider = _resolve_embedding_provider(config, forced_provider=forced_provider)
    if provider == "local":
        return _get_local_embeddings(config)

    gemini_key = (config.get("GEMINI_API_KEY") or "").strip()
    if not gemini_key:
        return None

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    model_name = _normalize_gemini_embedding_model(config.get("EMBEDDING_MODEL", "models/embedding-001"))
    return GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=gemini_key,
        task_type="retrieval_query",
    )


def _is_missing_gemini_embedding_model(error_message: str) -> bool:
    message = (error_message or "").lower()
    return "not found for api version" in message or "is not found" in message


class DocumentService:
    def __init__(self, config):
        self.config = config
        self.upload_folder = os.path.abspath(config["UPLOAD_FOLDER"])
        self.vector_db_folder = os.path.abspath(config["VECTOR_DB_FOLDER"])
        os.makedirs(self.upload_folder, exist_ok=True)
        os.makedirs(self.vector_db_folder, exist_ok=True)

        self.metadata_path = os.path.join(self.vector_db_folder, "metadata.json")
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Dict]:
        if not os.path.exists(self.metadata_path):
            return {}
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_metadata(self) -> None:
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=True, indent=2)

    def _compute_file_hash(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _split_text(self, text: str, source: str, doc_id: str, page: int = 0) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["CHUNK_SIZE"],
            chunk_overlap=self.config["CHUNK_OVERLAP"],
        )
        chunks = splitter.split_text(text or "")
        return [
            Document(
                page_content=chunk,
                metadata={
                    "source": source,
                    "doc_id": doc_id,
                    "page": page,
                    "chunk_index": idx,
                },
            )
            for idx, chunk in enumerate(chunks)
            if chunk.strip()
        ]

    def _build_documents_for_file(self, doc_id: str, meta: Dict) -> Tuple[List[Document], int]:
        file_path = meta["path"]
        filename = meta["filename"]
        suffix = Path(filename).suffix.lower()

        if suffix == ".pdf":
            pages = PyPDFLoader(file_path).load()
            docs: List[Document] = []
            for page_idx, page in enumerate(pages):
                docs.extend(self._split_text(page.page_content or "", filename, doc_id, page=page_idx))
            return docs, len(pages)

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        docs = self._split_text(text, filename, doc_id, page=0)
        return docs, 1

    def _vectorstore_dir(self, doc_id: str) -> str:
        return os.path.join(self.vector_db_folder, doc_id)

    def _read_document_sections(self, file_path: str, filename: str) -> Tuple[List[Tuple[int, str]], int]:
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf":
            pages = PyPDFLoader(file_path).load()
            sections = []
            for page_idx, page in enumerate(pages):
                text = (page.page_content or "").strip()
                if text:
                    sections.append((page_idx + 1, text))
            return sections, len(pages)

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = (f.read() or "").strip()
        sections = [(1, text)] if text else []
        return sections, 1

    def _resolve_document_path(self, doc_id: str, meta: Dict) -> Optional[str]:
        candidates: List[str] = []

        raw_path = meta.get("path")
        if raw_path:
            candidates.append(raw_path)

        stored_filename = (meta.get("stored_filename") or "").strip()
        filename = (meta.get("filename") or "").strip()
        record_id = str(meta.get("id") or doc_id)

        if stored_filename:
            candidates.append(os.path.join(self.upload_folder, stored_filename))

        if filename:
            candidates.append(os.path.join(self.upload_folder, filename))
            candidates.append(os.path.join(self.upload_folder, f"{record_id}_{filename}"))
            # Some legacy entries mixed UUID-with-dashes and UUID hex naming.
            candidates.append(os.path.join(self.upload_folder, f"{record_id.replace('-', '')}_{filename}"))

        seen = set()
        for candidate in candidates:
            if not candidate:
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            if os.path.exists(candidate):
                if meta.get("path") != candidate:
                    meta["path"] = candidate
                    self._save_metadata()
                return candidate
        return None

    def is_duplicate(self, temp_path: str) -> Optional[str]:
        file_hash = self._compute_file_hash(temp_path)
        for doc_id, meta in self._metadata.items():
            if meta.get("hash") == file_hash:
                return doc_id
        return None

    def get_document(self, doc_id: str) -> Optional[Dict]:
        return self._metadata.get(doc_id)

    def list_documents(self) -> List[Dict]:
        docs = []
        for meta in self._metadata.values():
            docs.append({k: v for k, v in meta.items() if k != "hash"})
        docs.sort(key=lambda d: d.get("uploaded_at", ""), reverse=True)
        return docs

    def save_document(self, temp_path: str, filename: str) -> Tuple[str, Dict]:
        doc_id = uuid.uuid4().hex
        final_name = f"{doc_id}_{filename}"
        final_path = os.path.join(self.upload_folder, final_name)
        os.replace(temp_path, final_path)

        now = datetime.now(timezone.utc).isoformat()
        file_hash = self._compute_file_hash(final_path)
        meta = {
            "id": doc_id,
            "filename": filename,
            "stored_filename": final_name,
            "path": final_path,
            "size": os.path.getsize(final_path),
            "chunk_count": 0,
            "page_count": 0,
            "status": "saved",
            "vectorised": False,
            "provider": None,
            "uploaded_at": now,
            "processed_at": None,
            "hash": file_hash,
        }
        self._metadata[doc_id] = meta
        self._save_metadata()
        return doc_id, meta

    def mark_document_ready_for_long_context(self, doc_id: str) -> Dict:
        meta = self._metadata.get(doc_id)
        if not meta:
            raise ValueError("Document not found")
        file_path = self._resolve_document_path(doc_id, meta)
        if not file_path:
            raise RuntimeError("Document file missing for this record. Please re-upload the document.")

        _, page_count = self._read_document_sections(file_path, meta.get("filename", ""))
        meta["chunk_count"] = 0
        meta["page_count"] = page_count
        meta["status"] = "processed"
        meta["vectorised"] = False
        meta["provider"] = "long_context"
        meta["processed_at"] = datetime.now(timezone.utc).isoformat()
        self._save_metadata()
        return meta

    def vectorise_document(self, doc_id: str) -> Dict:
        meta = self._metadata.get(doc_id)
        if not meta:
            raise ValueError("Document not found")
        file_path = self._resolve_document_path(doc_id, meta)
        if not file_path:
            raise RuntimeError("Document file missing for this record. Please re-upload the document.")

        embeddings, provider = _get_embeddings(self.config)

        docs, page_count = self._build_documents_for_file(doc_id, meta)
        if not docs:
            raise RuntimeError("No readable text found in document")

        store_dir = self._vectorstore_dir(doc_id)
        if os.path.exists(store_dir):
            shutil.rmtree(store_dir, ignore_errors=True)
        os.makedirs(store_dir, exist_ok=True)

        try:
            vectorstore = FAISS.from_documents(docs, embeddings)
        except Exception as exc:
            message = str(exc)
            provider_mode = (self.config.get("EMBEDDING_PROVIDER") or "auto").strip().lower()
            if provider == "gemini" and provider_mode == "auto" and _is_missing_gemini_embedding_model(message):
                local_embeddings = _get_local_embeddings(self.config)
                vectorstore = FAISS.from_documents(docs, local_embeddings)
                provider = "local"
            elif _is_missing_gemini_embedding_model(message):
                raise RuntimeError(
                    "Embedding model is unavailable. Set EMBEDDING_MODEL=models/embedding-001 "
                    "or set EMBEDDING_PROVIDER=local and retry."
                ) from exc
            else:
                raise
        vectorstore.save_local(store_dir)

        meta["chunk_count"] = len(docs)
        meta["page_count"] = page_count
        meta["status"] = "processed"
        meta["vectorised"] = True
        meta["provider"] = provider
        meta["processed_at"] = datetime.now(timezone.utc).isoformat()
        self._save_metadata()
        return meta

    def get_vectorstore(self, doc_id: str) -> Optional[FAISS]:
        meta = self._metadata.get(doc_id)
        if not meta or not meta.get("vectorised"):
            return None

        store_dir = self._vectorstore_dir(doc_id)
        if not os.path.exists(store_dir):
            return None

        provider = (meta.get("provider") or "").strip().lower()
        query_embeddings = _get_query_embeddings(self.config, forced_provider=provider or None)
        if query_embeddings is None:
            return None

        return FAISS.load_local(
            store_dir,
            query_embeddings,
            allow_dangerous_deserialization=True,
        )

    def get_combined_vectorstore(self, doc_ids: List[str]) -> Optional[FAISS]:
        stores = [self.get_vectorstore(doc_id) for doc_id in doc_ids]
        stores = [store for store in stores if store is not None]
        if not stores:
            return None

        merged = stores[0]
        for store in stores[1:]:
            merged.merge_from(store)
        return merged

    def get_long_context(self, doc_ids: List[str], max_chars: int = 300000) -> Tuple[str, List[Dict]]:
        context_parts: List[str] = []
        sources: List[Dict] = []
        seen_doc_ids = set()
        current_len = 0

        for doc_id in doc_ids:
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)

            meta = self._metadata.get(doc_id)
            if not meta:
                continue
            file_path = self._resolve_document_path(doc_id, meta)
            if not file_path:
                continue

            sections, _ = self._read_document_sections(file_path, meta.get("filename", ""))
            filename = meta.get("filename", "Unknown")

            for page_number, text in sections:
                if not text:
                    continue
                block = f"[Source: {filename}, Page: {page_number}]\n{text}"
                separator_len = 7 if context_parts else 0  # len("\n\n---\n\n")
                if max_chars > 0 and current_len + separator_len + len(block) > max_chars:
                    remaining = max_chars - current_len - separator_len
                    if remaining > 120:
                        context_parts.append(block[:remaining].rstrip() + "\n[CONTEXT TRUNCATED]")
                    return "\n\n---\n\n".join(context_parts), sources

                context_parts.append(block)
                current_len += separator_len + len(block)

                source = {
                    "filename": filename,
                    "page": page_number,
                    "chunk_index": 0,
                    "doc_id": doc_id,
                }
                if source not in sources:
                    sources.append(source)

        return "\n\n---\n\n".join(context_parts), sources

    def delete_document(self, doc_id: str) -> bool:
        meta = self._metadata.pop(doc_id, None)
        if not meta:
            return False

        try:
            file_path = self._resolve_document_path(doc_id, meta)
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            pass

        store_dir = self._vectorstore_dir(doc_id)
        if os.path.exists(store_dir):
            shutil.rmtree(store_dir, ignore_errors=True)

        self._save_metadata()
        return True
