import pytest
from unittest import mock
import numpy as np
from pathlib import Path
from auto_apply_bot.retrieval_interface.retrieval import LocalRagIndexer


@pytest.fixture
def mock_embedder():
    with mock.patch("auto_apply_bot.retrieval_interface.retrieval.SentenceTransformer") as MockModel:
        mock_model = MockModel.return_value
        mock_model.encode.side_effect = lambda texts, convert_to_numpy=True: np.ones((len(texts), 384))
        yield mock_model


@pytest.fixture
def rag_indexer(temp_project_dir, mock_embedder):
    return LocalRagIndexer(vector_store_dir_override=temp_project_dir)


@pytest.fixture
def test_data_dir():
    return Path(__file__).parent.parent / "data"


