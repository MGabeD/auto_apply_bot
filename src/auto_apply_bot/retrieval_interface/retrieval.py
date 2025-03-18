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
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from auto_apply_bot.logger import get_logger
from auto_apply_bot import resolve_project_source


logger = get_logger(__name__)
PROJECT_PATH = resolve_project_source()
ALLOWED_FILE_TYPES = [".txt", ".pdf", ".doc", ".docx"]

class LocalRagIndexer:
    allowed_file_types: str = ALLOWED_FILE_TYPES

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
        return ext in cls.allowed_file_types

    def _check_index_exists(self) -> bool:
        return (self.vector_store / "faiss_index.idx").exists() and (self.vector_store / "chunk_texts.json").exists()

    def load_document(self, filepath: Union[Path, str]) -> List[Document]:
        ext = os.path.splitext(filepath)[1]
        if ext == ".pdf":
            loader = PyPDFLoader(filepath)
        elif ext == ".docx":
            loader = Docx2txtLoader(filepath)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
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

    def _chunk_documents(self, documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")
        return [chunk.page_context for chunk in chunks]

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