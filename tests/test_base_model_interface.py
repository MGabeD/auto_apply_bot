import pytest
from auto_apply_bot.model_interfaces.base_model_interface import BaseModelInterface


def test_tokenizer_loaded(patch_model_interface):
    interface = BaseModelInterface()
    interface._load_tokenizer()
    assert interface.tokenizer is not None


def test_pipeline_loads_after_model(patch_model_interface):
    interface = BaseModelInterface()
    interface._load_tokenizer()
    interface._load_model()
    interface._load_pipeline()
    assert interface.pipe is not None


def test_context_manager_initializes_and_cleans(patch_model_interface):
    interface = BaseModelInterface()
    with interface as model_interface:
        assert model_interface.tokenizer is not None
        assert model_interface.model is not None
        assert model_interface.pipe is not None
    assert interface.pipe is None
    assert interface.model is None


def test_run_prompts_strips_prompt_text(patch_model_interface):
    interface = BaseModelInterface()
    with interface:
        prompts = ["This is prompt 1", "Prompt 2 here"]
        outputs = interface.run_prompts(prompts)
        assert outputs == ["output", "output"]


def test_run_prompts_with_post_process(patch_model_interface):
    interface = BaseModelInterface()
    with interface:
        prompts = ["Example"]
        outputs = interface.run_prompts(prompts, post_process_fn=lambda x: x[::-1])
        assert outputs == ["tuptuo elpmaxE"]
