from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredWordDocumentLoader
from typing import List, Union, Optional
from pathlib import Path
from langchain.schema import Document
from auto_apply_bot.logger import get_logger


logger = get_logger(__name__)


LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": lambda path: TextLoader(path, encoding="utf-8"),
    ".docx": Docx2txtLoader,
    ".doc": UnstructuredWordDocumentLoader,
}


def load_documents(file_paths: List[Union[str, Path]], loader_map_override: Optional[dict] = None) -> List[Document]:
    """
    Loads LangChain Document objects from multiple file paths.
    """
    loader_map = loader_map_override or LOADER_MAP
    all_docs = []
    for file_path in file_paths:
        path = Path(file_path)
        ext = path.suffix.lower()

        loader_class = loader_map.get(ext)
        if not loader_class:
            raise ValueError(f"Unsupported file type: {ext}")

        loader = loader_class(str(path))
        docs = loader.load()
        all_docs.extend(docs)
        logger.info(f"Loaded {len(docs)} documents from {path}")
    return all_docs
