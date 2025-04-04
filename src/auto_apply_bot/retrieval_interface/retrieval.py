import hashlib
import os
import json
from typing import Union, List, Optional
from pathlib import Path
from xml.dom.minidom import Document
import numpy as np
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from auto_apply_bot.utils.logger import get_logger
from auto_apply_bot import resolve_component_dirs_path, LOADER_MAP
from auto_apply_bot.utils.loader import load_documents
from contextlib import contextmanager
import torch


logger = get_logger(__name__)
VECTOR_STORE_DIR = resolve_component_dirs_path("vector_store")


class LocalRagIndexer:
    loader_map: dict = LOADER_MAP

    def __init__(self, 
                 vector_store_dir_override: Union[str,Path] = VECTOR_STORE_DIR, 
                 embed_model_name: str = "BAAI/bge-large-en-v1.5", 
                 lazy_embedder: bool = False):
        self.vector_store_dir: Path = vector_store_dir_override
        self.embedder_model_name: str = embed_model_name
        self.lazy_embedder: bool = lazy_embedder
        self._embedder: Optional[SentenceTransformer] = None if lazy_embedder else SentenceTransformer(embed_model_name)
        self.index: Union[faiss.IndexFlatL2, None] = None
        self.chunk_texts: List[str] = []
        self.chunk_hashes: set = set()

        if self._check_index_exists():
            self.load()
        else:
            logger.warning("No existing vector store found. RAG is empty but still functional")

    @classmethod
    def is_allowed_file_type(cls, filename: str) -> bool:
        """
        Checks if a file type is supported.
        :param filename: Path to the file
        :return: True if the file type is supported, False otherwise
        """
        ext = os.path.splitext(filename)[1].lower()
        return ext in cls.loader_map.keys()

    @classmethod
    def get_supported_file_types(cls) -> List[str]:
        """
        Returns a list of supported file types.
        :return: List of supported file types
        """
        return sorted(cls.loader_map.keys())

    @contextmanager
    def _get_embedder(self):
        """
        Context manager to return an embedder depending on lazy or persistent mode.
        """
        if self.lazy_embedder:
            logger.info("Instantiating embedder in lazy mode")
            embedder = SentenceTransformer(self.embedder_model_name)
            try:
                yield embedder
            finally:
                del embedder
                torch.cuda.empty_cache()
                logger.info("Lazy embedder cleaned up and CUDA cache cleared")
        else:
            if self._embedder is None:
                logger.info(f"Persistent embedder was cleared, re-initializing on demand")
                self._embedder = SentenceTransformer(self.embedder_model_name)
            yield self._embedder

    def cleanup(self):
        """
        Cleans up the RAG indexer.
        Currently does nothing.
        """
        return

    def clear_embedder(self):
        """
        Clears the persistent embedder.
        """
        if self.lazy_embedder:
            logger.warning("clear_embedder called in lazy mode, this will have no effect")
            return
        if self._embedder is not None:
            del self._embedder
            self._embedder = None
            torch.cuda.empty_cache()
            logger.info("Persistent embedder cleared")
        else:
            logger.warning("No persistent embedder to clear")

    def _check_index_exists(self) -> bool:
        """
        Checks if the RAG index exists.
        :return: True if the index exists, False otherwise
        """
        return (self.vector_store_dir / "faiss_index.idx").exists() and (self.vector_store_dir / "chunk_texts.json").exists()

    def add_documents(self, file_paths: Union[List[str], List[Path]]) -> None:
        """
        Adds documents to the RAG index.
        :param file_paths: List of file paths to add
        """
        all_docs = load_documents(file_paths, self.loader_map)

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
        """
        Chunks documents into smaller chunks of text.
        :param documents: List of Document objects
        :param chunk_size: Maximum size of each chunk
        :param chunk_overlap: Overlap between chunks
        :return: List of chunked text
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")
        return [chunk.page_content for chunk in chunks]

    def _embed_chunks(self, chunks: List[str], batch_size: int = 16) -> 'np.ndarray':
        """
        Embeds chunks of text using SentenceTransformer.
        :param chunks: List of text chunks to embed
        :param batch_size: Number of chunks to embed in each batch
        :return: numpy array of embedded chunks
        """
        embeddings_list = []

        with self._get_embedder() as embedder:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_embeddings = embedder.encode(batch, convert_to_numpy=True)
                embeddings_list.append(batch_embeddings)

        logger.info(f"Embedded {len(chunks)} chunks in batches of {batch_size}")
        return np.vstack(embeddings_list)

    def _filter_duplicates(self, chunks: List[str]) -> List[str]:
        """
        Filter out duplicate chunks based on their hash.
        :param chunks: List of text chunks to filter
        :return: List of unique chunks
        """
        new_chunks= []
        for chunk in chunks:
            chunk_hash = hashlib.sha256(chunk.encode('utf-8')).hexdigest()
            if chunk_hash not in self.chunk_hashes:
                self.chunk_hashes.add(chunk_hash)
                new_chunks.append(chunk)
        logger.info(f"Filtered {len(chunks) - len(new_chunks)} duplicated chunks")
        return new_chunks

    def save(self) -> None:
        """
        Saves the index and metadata.
        """
        faiss.write_index(self.index, str(self.vector_store_dir / "faiss_index.idx"))
        with open(self.vector_store_dir / "chunk_texts.json", "w") as f:
            json.dump(self.chunk_texts, f)
        with open(self.vector_store_dir / "chunk_hashes.json", "w") as f:
            json.dump(list(self.chunk_hashes), f)
        logger.info(f"Index and metadata saved to {self.vector_store_dir}")

    def load(self) -> None:
        """
        Loads the index and metadata.
        """
        self.index = faiss.read_index(str(self.vector_store_dir / "faiss_index.idx"))
        with open(self.vector_store_dir / "chunk_texts.json", "r") as f:
            self.chunk_texts = json.load(f)
        with open(self.vector_store_dir / "chunk_hashes.json", "r") as f:
            self.chunk_hashes = set(json.load(f))
        logger.info(f"Index and metadata loaded from {self.vector_store_dir}")

    def query(self,
              query_text: str,
              top_k: int = 5,
              similarity_threshold: float = 0.0,
              deduplicate: bool = False, ) -> List[dict]:
        """
        Runs a single query over the RAG index.
        :param query_text: Query string
        :param top_k: Number of results to return
        :param similarity_threshold: Minimum similarity score to include a result
        :param deduplicate: Whether to deduplicate results within each query (based on text hash)
        :return: List of dicts
        """
        if self.index is None or not self.chunk_texts:
            raise ValueError("RAG index is empty. Add documents before querying.")

        with self._get_embedder() as embedder:
            query_embedding = embedder.encode([query_text], convert_to_numpy=True)

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
    
    def batch_query(self,
                    query_texts: List[str],
                    top_k: int = 5,
                    similarity_threshold: float = 0.0,
                    deduplicate: bool = False, ) -> dict[str, List[dict]]:
        """
        Runs batched quereies over multiple query strings.
        :param query_texts: List of query strings
        :param top_k: Number of results to return per query
        :param similarity_threshold: Minimum similarity score to include a result
        :param deduplicate: Whether to deduplicate results within each query (based on text hash)
        :return: dict of results per query, each as a list of dicts
        """
        if self.index is None or not self.chunk_texts:
            raise ValueError("RAG index is empty. Add documents before querying.")
        
        with self._get_embedder() as embedder:
            query_embeddings = embedder.encode(query_texts, convert_to_numpy=True)

        distances, indices = self.index.search(query_embeddings, top_k)

        batch_results = {}
        for i, query in enumerate(query_texts):
            max_dist = np.max(distances[i]) if np.max(distances[i]) > 0 else 1.0
            similarities = 1 - (distances[i] / max_dist)

            results = []
            seen_texts = set()
            for idx, sim, raw_dist in zip(indices[i], similarities, distances[i]):
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
                    "length": len(chunk_text), 
                })
            batch_results[query] = results
        return batch_results

    def wipe_rag(self) -> None:
        """
        Completely wipes the in-memory and on-disk RAG data.
        """
        self.index = None
        self.chunk_texts = []
        self.chunk_hashes = set()

        try:
            index_path = self.vector_store_dir / "faiss_index.idx"
            chunks_path = self.vector_store_dir / "chunk_texts.json"
            hashes_path = self.vector_store_dir / "chunk_hashes.json"
            for file_path in [index_path, chunks_path, hashes_path]:
                if file_path.exists():
                    os.remove(file_path)
                    logger.info(f"Deleted: {file_path}")
            logger.warning("RAG has been wiped completely.")
        except Exception as e:
            logger.error(f"Error wiping RAG files: {e}", exc_info=True)