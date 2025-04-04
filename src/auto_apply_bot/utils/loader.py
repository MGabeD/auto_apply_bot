from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredWordDocumentLoader
from typing import List, Union, Optional
from pathlib import Path
from langchain.schema import Document
from auto_apply_bot.utils.logger import get_logger
from auto_apply_bot import LOADER_MAP


logger = get_logger(__name__)


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


def load_texts_from_files(file_paths: List[Union[str, Path]]) -> List[str]:
    """
    Loads documents using load_documents and extracts clean page_content strings.
    """
    docs = load_documents(file_paths)
    logger.info(f"Loaded {len(docs)} documents")
    return [doc.page_content.strip() for doc in docs if doc.page_content.strip()]