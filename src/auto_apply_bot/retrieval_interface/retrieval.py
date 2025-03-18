import hashlib
import os
import json
from typing import Union, List
from pathlib import Path
from xml.dom.minidom import Document
import numpy as np
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredWordDocumentLoader
from auto_apply_bot.logger import get_logger
from auto_apply_bot import resolve_project_source


logger = get_logger(__name__)
PROJECT_PATH = resolve_project_source()


LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": lambda path: TextLoader(path, encoding="utf-8"),
    ".docx": Docx2txtLoader,
    ".doc": UnstructuredWordDocumentLoader,
}


class LocalRagIndexer:
    loader_map: dict = LOADER_MAP

    def __init__(self, project_dir: Union[str,Path] = PROJECT_PATH, embed_model_name: str = "all-MiniLM-L6-v2"):
        self.project_dir: Path = Path(project_dir)
        self.vector_store: Path = (project_dir / "vector_store")
        self.embedder = SentenceTransformer(embed_model_name)
        self.index: Union[faiss.IndexFlatL2, None] = None
        self.chunk_texts: List[str] = []
        self.chunk_hashes: set = set()
        os.makedirs(self.vector_store, exist_ok=True)

        if self._check_index_exists():
            self.load()
        else:
            logger.warning("No existing vector store found. RAG is empty but still functional")

    @classmethod
    def is_allowed_file_type(cls, filename: str) -> bool:
        ext = os.path.splitext(filename)[1].lower()
        return ext in cls.loader_map.keys()

    @classmethod
    def get_supported_file_types(cls) -> List[str]:
        return sorted(cls.loader_map.keys())

    def _check_index_exists(self) -> bool:
        return (self.vector_store / "faiss_index.idx").exists() and (self.vector_store / "chunk_texts.json").exists()

    def load_document(self, filepath: Union[Path, str]) -> List[Document]:
        ext = os.path.splitext(filepath)[1].lower()

        loader_class = self.loader_map.get(ext)
        if not loader_class:
            raise ValueError(f"Unsupported file type: {ext}")

        loader = loader_class(filepath) if not callable(loader_class(filepath)) else loader_class(filepath)
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} pages from {filepath}")
        return docs

    def add_documents(self, file_paths: Union[List[str], List[Path]]) -> None:
        all_docs = []
        for path in file_paths:
            docs = self.load_document(path)
            all_docs.extend(docs)

        chunks = self._chunk_documents(all_docs)
        new_chunks = self._filter_duplicates(chunks)
        if not new_chunks:
            logger.warning("No new unique chunks to add.")
            return

        embeddings = self._embed_chunks(new_chunks)

        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)
        self.chunk_texts.extend(new_chunks)
        logger.info(f"Added {len(new_chunks)} unique chunks to the index.")
        self.save()

    def _chunk_documents(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 50) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")
        return [chunk.page_content for chunk in chunks]

    def _embed_chunks(self, chunks: List[str], batch_size: int = 16) -> 'np.ndarray':
        embeddings_list = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_embeddings = self.embedder.encode(batch, convert_to_numpy=True)
            embeddings_list.append(batch_embeddings)
        logger.info(f"Embedded {len(chunks)} chunks in batches of {batch_size}")
        return np.vstack(embeddings_list)

    def _filter_duplicates(self, chunks: List[str]) -> List[str]:
        new_chunks= []
        for chunk in chunks:
            chunk_hash = hashlib.sha256(chunk.encode('utf-8')).hexdigest()
            if chunk_hash not in self.chunk_hashes:
                self.chunk_hashes.add(chunk_hash)
                new_chunks.append(chunk)
        logger.info(f"Filtered {len(chunks) - len(new_chunks)} duplicated chunks")
        return new_chunks

    def save(self) -> None:
        faiss.write_index(self.index, str(self.vector_store / "faiss_index.idx"))
        with open(self.vector_store / "chunk_texts.json", "w") as f:
            json.dump(self.chunk_texts, f)
        with open(self.vector_store / "chunk_hashes.json", "w") as f:
            json.dump(list(self.chunk_hashes), f)
        logger.info(f"Index and metadata saved to {self.vector_store}")

    def load(self) -> None:
        self.index = faiss.read_index(str(self.vector_store / "faiss_index.idx"))
        with open(self.vector_store / "chunk_texts.json", "r") as f:
            self.chunk_texts = json.load(f)
        with open(self.vector_store / "chunk_hashes.json", "r") as f:
            self.chunk_hashes = set(json.load(f))
        logger.info(f"Index and metadata loaded from {self.vector_store}")

    def query(self,
              query_text: str,
              top_k: int = 5,
              similarity_threshold: float = 0.0,
              deduplicate: bool = False, ) -> List[dict]:
        if self.index is None or not self.chunk_texts:
            raise ValueError("RAG index is empty. Add documents before querying.")

        query_embedding = self.embedder.encode([query_text], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        max_dist = np.max(distances) if np.max(distances) > 0 else 1.0
        similarities = 1 - (distances[0] / max_dist)

        results = []
        seen_texts = set()
        for idx, sim, raw_dist in zip(indices[0], similarities, distances[0]):
            if idx >= len(self.chunk_texts):
                continue
            chunk_text = self.chunk_texts[idx]
            if sim < similarity_threshold:
                continue
            if deduplicate:
                text_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                if text_hash in seen_texts:
                    continue
                seen_texts.add(text_hash)
            results.append({
                "text": chunk_text,
                "similarity": round(float(sim), 4),
                "distance": round(float(raw_dist), 4),
                "length": len(chunk_text)
            })
        logger.info(f"Query returned {len(results)} results for: {query_text}")
        return results

    def wipe_rag(self) -> None:
        """Completely wipes the in-memory and on-disk RAG data."""
        self.index = None
        self.chunk_texts = []
        self.chunk_hashes = set()

        try:
            index_path = self.vector_store / "faiss_index.idx"
            chunks_path = self.vector_store / "chunk_texts.json"
            hashes_path = self.vector_store / "chunk_hashes.json"
            for file_path in [index_path, chunks_path, hashes_path]:
                if file_path.exists():
                    os.remove(file_path)
                    logger.info(f"Deleted: {file_path}")
            logger.warning("RAG has been wiped completely.")
        except Exception as e:
            logger.error(f"Error wiping RAG files: {e}", exc_info=True)