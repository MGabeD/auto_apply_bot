import pytest
from unittest.mock import MagicMock
from auto_apply_bot.model_interfaces.cover_letter_generator.cover_letter_generator import CoverLetterModelInterface, DialoguePairDataset


def test_add_and_reset_feedback():
    model = CoverLetterModelInterface()
    
    count = model.add_feedback_example("Prompt text", "Response text")
    assert count == 1
    assert len(model._feedback_examples) == 1

    assert model.add_feedback_example("", "") is None
    assert len(model._feedback_examples) == 1  

    model.reset_feedback()
    assert len(model._feedback_examples) == 0


def test_generate_cover_letter_calls_run_prompts(monkeypatch):
    model = CoverLetterModelInterface()
    mock_run = MagicMock(return_value=["Generated Letter"])
    monkeypatch.setattr(model, "run_prompts", mock_run)

    job_desc = "Great job opening."
    resume_snippets = ["Built a rocket.", "Wrote a parser."]

    result = model.generate_cover_letter(job_desc, resume_snippets)
    assert "Generated Letter" in result

    expected_prompt_part = "Job Description:\nGreat job opening."
    mock_run.assert_called_once()
    assert expected_prompt_part in mock_run.call_args[0][0][0]


def test_assess_experience_calls_run_prompts(monkeypatch):
    model = CoverLetterModelInterface()
    mock_run = MagicMock(return_value=["Assessment result"])
    monkeypatch.setattr(model, "run_prompts", mock_run)

    job_desc = "Must know Python."
    experience = ["Built web apps", "Used Flask"]

    result = model.assess_experience_against_posting(job_desc, experience)
    assert "Assessment result" in result
    assert "Job Description:\nMust know Python." in mock_run.call_args[0][0][0]


def test_train_on_dialogue_pairs_with_buffer(dummy_tokenizer, monkeypatch):
    model = CoverLetterModelInterface()
    model.tokenizer = dummy_tokenizer
    model.ensure_lora_adapter_loaded = lambda error_message="": None
    model.fine_tune = lambda train_dataset, output_subdir_override=None: "some_path"

    model.add_feedback_example("Prompt A", "Response A")
    path = model.train_on_dialogue_pairs(load_from_buffer=True)
    assert path == "some_path"
    assert len(model._feedback_examples) == 0


def test_train_on_existing_letters(monkeypatch, dummy_tokenizer):
    monkeypatch.setattr(
        "auto_apply_bot.model_interfaces.cover_letter_generator.cover_letter_generator.load_texts_from_files",
        lambda paths: ["Letter 1", "Letter 2"]
    )

    model = CoverLetterModelInterface()
    model.tokenizer = dummy_tokenizer
    model.ensure_lora_adapter_loaded = lambda error_message="": None
    model.fine_tune = lambda train_dataset, output_subdir_override=None: "trained_path"

    result = model.train_on_existing_letters(["/fake/path.txt"])
    assert result == "trained_path"


def test_exit_triggers_training(monkeypatch):
    model = CoverLetterModelInterface()
    model._feedback_examples = [("Prompt", "Response")]
    monkeypatch.setattr(model, "train_on_dialogue_pairs", lambda load_from_buffer=False, **kwargs: "trained")

    result = model.__exit__(None, None, None)
    assert result is None  


def test_dialogue_pair_dataset_len_and_getitem(dummy_tokenizer):
    pairs = [("Prompt 1", "Resp 1"), ("Prompt 2", "Resp 2")]
    dataset = DialoguePairDataset(pairs, dummy_tokenizer)

    assert len(dataset) == 2
    item = dataset[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item