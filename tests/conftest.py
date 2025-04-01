import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path
from unittest import mock
from auto_apply_bot.retrieval_interface.retrieval import LocalRagIndexer
from auto_apply_bot.model_interfaces.skill_parser import SkillParser
from tests.mocks import mock_loader


@pytest.fixture
def mock_embedder():
    with mock.patch("auto_apply_bot.retrieval_interface.retrieval.SentenceTransformer") as MockModel:
        mock_model = MockModel.return_value
        mock_model.encode.side_effect = lambda texts, convert_to_numpy=True: np.ones((len(texts), 384))
        yield mock_model


@pytest.fixture
def temp_project_dir():
    dirpath = tempfile.mkdtemp()
    yield Path(dirpath)
    shutil.rmtree(dirpath)


@pytest.fixture
def rag_indexer(temp_project_dir, mock_embedder):
    return LocalRagIndexer(project_dir=temp_project_dir)


@pytest.fixture
def test_data_dir():
    return Path(__file__).parent / "data"


@pytest.fixture
def mock_parser(monkeypatch):
    parser = SkillParser()
    parser.pipe = mock.Mock()
    return parser


@pytest.fixture
def dummy_tokenizer():
    return mock_loader.DummyTokenizer()


@pytest.fixture
def dummy_model():
    return mock_loader.DummyModel()


@pytest.fixture
def dummy_pipeline():
    return mock_loader.DummyPipeline()


@pytest.fixture
def patch_model_interface(monkeypatch):
    mock_loader.patch_model_and_tokenizer(monkeypatch)