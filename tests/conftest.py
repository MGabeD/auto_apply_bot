import pytest
import tempfile
import shutil
from pathlib import Path
from unittest import mock
from auto_apply_bot.model_interfaces.skill_parser import SkillParser
from tests.mocks import mock_loader
from auto_apply_bot.model_interfaces.lora_model_interface import LoraModelInterface


@pytest.fixture
def temp_project_dir():
    dirpath = tempfile.mkdtemp()
    yield Path(dirpath)
    shutil.rmtree(dirpath)


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
def patch_model_interface(monkeypatch, temp_project_dir):
    mock_loader.patch_model_and_tokenizer(monkeypatch)
    monkeypatch.setattr(
        LoraModelInterface,
        "_get_latest_adapter_path",
        lambda self: None
    )
    original_init = LoraModelInterface.__init__
    def custom_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.lora_weights_dir = temp_project_dir / "lora_weights"
        self.lora_weights_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(LoraModelInterface, "__init__", custom_init)

@pytest.fixture
def dummy_peft_model():
    return mock_loader.DummyPeftModel()