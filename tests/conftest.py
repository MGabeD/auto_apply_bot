import pytest
import tempfile
import shutil
from pathlib import Path
from tests.mocks import mock_loader
import os


def pytest_addoption(parser):
    parser.addoption("--quiet-logs", action="store_true", default=False, help="Disable logging output from the application during test")


def pytest_configure(config):
    if config.getoption("--quiet-logs"):
        os.environ["QUIET_MODE"] = "true"


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