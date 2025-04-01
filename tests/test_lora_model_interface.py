import pytest
from auto_apply_bot.model_interfaces.lora_model_interface import LoraModelInterface
from tests.mocks import mock_loader


def test_enter_exit_lora_model(patch_model_interface):
    model = LoraModelInterface()
    with model as m:
        assert m.tokenizer is not None
        assert m.model is not None
        assert m.pipe is not None
    assert model.pipe is None
    assert model.base_model is None


def test_freeze_lora_adapter_sets_requires_grad_false(patch_model_interface):
    from tests.mocks.mock_loader import DummyPeftModel
    model = LoraModelInterface()

    with model:
        dummy_peft = DummyPeftModel()
        model.model = dummy_peft
        model.last_loaded_adapter_path = model.lora_weights_dir / "mock_adapter"

        model.freeze_lora_adapter()

        for param in model.model.parameters():
            assert param.requires_grad is False


def test_unfreeze_lora_adapter_sets_requires_grad_true(patch_model_interface):
    from tests.mocks.mock_loader import DummyPeftModel
    model = LoraModelInterface()

    with model:
        dummy_peft = DummyPeftModel()
        for param in dummy_peft.parameters():
            param.requires_grad = False
        model.model = dummy_peft
        model.last_loaded_adapter_path = model.lora_weights_dir / "mock_adapter"

        model.unfreeze_lora_adapter()

        for param in model.model.parameters():
            assert param.requires_grad is True


def test_repr_contains_flags(patch_model_interface):
    model = LoraModelInterface()
    with model:
        out = repr(model)
    assert "adapter_loaded=" in out
    assert "adapter_frozen=" in out


def test_reset_lora_adapter_sets_model_to_base(patch_model_interface):
    model = LoraModelInterface()
    with model:
        model.reset_lora_adapter()
        assert model.model == model.base_model
        assert model.last_loaded_adapter_path is None


def test_freeze_lora_adapter_raises_if_no_adapter_loaded(patch_model_interface):
    model = LoraModelInterface()
    with model:
        model.model = model.base_model  # not a PeftModel
        model.last_loaded_adapter_path = None
        with pytest.raises(RuntimeError, match="No LoRA adapter is currently loaded."):
            model.freeze_lora_adapter()


def test_unfreeze_lora_adapter_raises_if_no_adapter_loaded(patch_model_interface):
    model = LoraModelInterface()
    with model:
        model.model = model.base_model
        model.last_loaded_adapter_path = None
        with pytest.raises(RuntimeError, match="No LoRA adapter is currently loaded."):
            model.unfreeze_lora_adapter()


def test_reset_lora_adapter_raises_if_base_model_none(patch_model_interface):
    model = LoraModelInterface()
    with model:
        model.base_model = None
        with pytest.raises(RuntimeError, match="Base model must be loaded before resetting LoRA adapter."):
            model.reset_lora_adapter()


def test_has_loaded_lora_adapter_false_without_adapter(patch_model_interface):
    model = LoraModelInterface()
    with model:
        model.model = model.base_model  # not a PeftModel
        model.last_loaded_adapter_path = None
        assert model.has_loaded_lora_adapter() is False


def test_get_adapter_config_structure(patch_model_interface):
    model = LoraModelInterface()
    with model:
        config = model.get_adapter_config()
        assert isinstance(config, dict)
        assert set(config.keys()) == {
            "loaded_adapter", "last_trained_adapter", "adapter_frozen", "available_adapters"
        }


def test_list_available_adapters_returns_empty_when_none(patch_model_interface):
    model = LoraModelInterface()
    with model:
        assert model.list_available_adapters() == []


def test_adapter_frozen_is_none_without_adapter(patch_model_interface):
    model = LoraModelInterface()
    with model:
        model.model = model.base_model
        model.last_loaded_adapter_path = None
        assert model.adapter_frozen is None
