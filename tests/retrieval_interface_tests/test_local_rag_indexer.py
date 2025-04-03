import pytest
import numpy as np
from unittest import mock
from auto_apply_bot.retrieval_interface.retrieval import LocalRagIndexer
from tests.mocks.mock_loader import DummyLoader
from auto_apply_bot.utils.logger import get_logger


logger = get_logger(__name__)


# MARK: since this is a for fun project these tests are super super basic, don't cover all cases, and are just to give me an indicator 
#  for/if/when I break one of my interfaces unintentionally


def test_is_allowed_file_type():
    assert LocalRagIndexer.is_allowed_file_type("file.pdf")
    assert not LocalRagIndexer.is_allowed_file_type("file.csv")


def test_get_supported_file_types():
    supported = LocalRagIndexer.get_supported_file_types()
    assert ".pdf" in supported
    assert ".doc" in supported


def test_check_index_exists_false(rag_indexer):
    assert not rag_indexer._check_index_exists()


def test_check_index_exists_true(rag_indexer):
    # Simulate saved index files
    (rag_indexer.vector_store_dir / "faiss_index.idx").touch()
    (rag_indexer.vector_store_dir / "chunk_texts.json").touch()
    assert rag_indexer._check_index_exists()


def test_chunk_documents(monkeypatch, rag_indexer):
    dummy_doc = mock.Mock()
    dummy_doc.page_content = "some text"
    splitter_mock = mock.Mock()
    splitter_mock.split_documents.return_value = [dummy_doc, dummy_doc]

    monkeypatch.setattr("auto_apply_bot.retrieval_interface.retrieval.RecursiveCharacterTextSplitter", lambda **kwargs: splitter_mock)
    chunks = rag_indexer._chunk_documents([dummy_doc])
    assert len(chunks) == 2


def test_filter_duplicates(rag_indexer):
    chunks = ["chunk A", "chunk B", "chunk A"]  # 1 duplicate
    unique_chunks = rag_indexer._filter_duplicates(chunks)
    assert unique_chunks == ["chunk A", "chunk B"]
    assert len(rag_indexer.chunk_hashes) == 2


def test_embed_chunks(rag_indexer):
    chunks = ["text1", "text2"]
    emb = rag_indexer._embed_chunks(chunks)
    assert emb.shape == (2, 384)


def test_add_documents(monkeypatch, rag_indexer):
    monkeypatch.setitem(rag_indexer.loader_map, ".txt", DummyLoader)
    dummy_file = rag_indexer.vector_store_dir / "dummy.txt"
    dummy_file.write_text("dummy")
    rag_indexer.add_documents([dummy_file])
    assert rag_indexer.index is not None
    assert rag_indexer.chunk_texts


def test_save_and_load(rag_indexer):
    rag_indexer.chunk_texts = ["abc"]
    rag_indexer.chunk_hashes = {"hash1"}
    rag_indexer.index = mock.Mock()
    rag_indexer.index.ntotal = 1

    with mock.patch("auto_apply_bot.retrieval_interface.retrieval.faiss.write_index") as write_mock:
        rag_indexer.save()
        write_mock.assert_called_once()

    with mock.patch("auto_apply_bot.retrieval_interface.retrieval.faiss.read_index", return_value=mock.Mock()):
        rag_indexer.load()
        assert rag_indexer.chunk_texts == ["abc"]
        assert "hash1" in rag_indexer.chunk_hashes


def test_query_success(rag_indexer):
    rag_indexer.chunk_texts = ["A", "B"]
    rag_indexer.index = mock.Mock()
    rag_indexer.index.search.return_value = (np.array([[0.1, 0.2]]), np.array([[0, 1]]))
    result = rag_indexer.query("query")
    assert len(result) == 2
    assert "text" in result[0]


def test_query_empty_index(rag_indexer):
    with pytest.raises(ValueError):
        rag_indexer.query("query")


def test_batch_query_success(rag_indexer):
    rag_indexer.chunk_texts = ["A", "B"]
    rag_indexer.index = mock.Mock()
    rag_indexer.index.search.return_value = (np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([[0, 1], [0, 1]]))

    batch = rag_indexer.batch_query(["query1", "query2"])
    assert len(batch) == 2
    assert all(isinstance(v, list) for v in batch.values())


def test_wipe_rag(rag_indexer):
    rag_indexer.chunk_texts = ["abc"]
    rag_indexer.chunk_hashes = {"hash"}
    rag_indexer.index = mock.Mock()
    (rag_indexer.vector_store_dir / "faiss_index.idx").touch()
    rag_indexer.wipe_rag()
    assert rag_indexer.chunk_texts == []
    assert rag_indexer.index is None


def test_batch_query_full_rag(rag_indexer, test_data_dir):
    file_paths = list(test_data_dir.glob('*'))
    assert file_paths, "No test files found in tests/data/"

    rag_indexer.add_documents(file_paths=file_paths)
    assert rag_indexer.chunk_texts, "No chunks added to RAG"
    assert rag_indexer.index is not None

    queries = [
        "What is experience with python does this candidate have?",
        "Extract key skills discussed in the text.",
        "What is the candidate's name?",
    ]
    
    results = rag_indexer.batch_query(query_texts=queries, top_k=5)
    assert isinstance(results, dict)
    assert set(results.keys()) == set(queries)
    logger.info(results)

    for query, query_results in results.items():
        assert isinstance(query_results, list)
        for result in query_results:
            assert "text" in result
            assert "similarity" in result

    rag_indexer.wipe_rag()

    assert rag_indexer.index is None
    assert rag_indexer.chunk_texts == []
    assert rag_indexer.chunk_hashes == set()

    vector_store = rag_indexer.vector_store_dir
    assert not (vector_store / "faiss_index.idx").exists()
    assert not (vector_store / "chunk_texts.json").exists()
    assert not (vector_store / "chunk_hashes.json").exists()

