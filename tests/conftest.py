import pytest
import tempfile
import shutil
from pathlib import Path
from tests.mocks import mock_loader


@pytest.fixture
def temp_project_dir():
    dirpath = tempfile.mkdtemp()
    yield Path(dirpath)
    shutil.rmtree(dirpath)


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
def dummy_peft_model():
    return mock_loader.DummyPeftModel()