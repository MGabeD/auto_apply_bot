import pytest
from unittest import mock
from auto_apply_bot.model_interfaces.skill_parser import SkillParser
from auto_apply_bot.model_interfaces.lora_model_interface import LoraModelInterface
from tests.mocks import mock_loader

@pytest.fixture
def mock_parser(monkeypatch):
    parser = SkillParser()
    parser.pipe = mock.Mock()
    return parser


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