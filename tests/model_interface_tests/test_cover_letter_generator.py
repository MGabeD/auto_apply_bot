import pytest
from unittest.mock import MagicMock
from auto_apply_bot.model_interfaces.cover_letter_generator.cover_letter_generator import CoverLetterModelInterface, DialoguePairDataset


@pytest.mark.parametrize("prompt, response, expected_delta", [
    ("Prompt 1", "Response 1", 1),
    ("", "Response", 0),
    ("Prompt", "", 0),
    (None, "Response", 0),
    ("Prompt", None, 0)
])
def test_add_feedback_example_param(prompt, response, expected_delta):
    model = CoverLetterModelInterface()
    initial = len(model._feedback_examples)
    model.add_feedback_example(prompt, response)
    final = len(model._feedback_examples)
    assert final - initial == expected_delta


def test_reset_feedback_clears_examples():
    model = CoverLetterModelInterface()
    model.add_feedback_example("Prompt", "Response")
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


@pytest.mark.parametrize("dialogue_pairs, load_from_buffer, expect_value_error, expect_none", [
    (None, False, True, False),       # neither input nor buffer
    ([], True, False, True),          # empty buffer
    ([("Prompt", "Response")], False, False, False)  # valid data
])
def test_train_on_dialogue_pairs_param(dummy_tokenizer, dialogue_pairs, load_from_buffer, expect_value_error, expect_none):
    model = CoverLetterModelInterface()
    model.tokenizer = dummy_tokenizer
    model.ensure_lora_adapter_loaded = lambda error_message="": None
    model.fine_tune = lambda train_dataset, output_subdir_override=None: "trained"
    if load_from_buffer:
        model._feedback_examples = dialogue_pairs or []
    if expect_value_error:
        with pytest.raises(ValueError):
            model.train_on_dialogue_pairs(dialogue_pairs=dialogue_pairs, load_from_buffer=load_from_buffer)
    else:
        result = model.train_on_dialogue_pairs(dialogue_pairs=dialogue_pairs, load_from_buffer=load_from_buffer)
        if expect_none:
            assert result is None
        else:
            assert result == "trained"


def test_train_on_dialogue_pairs_raises_if_no_tokenizer():
    model = CoverLetterModelInterface()
    model.ensure_lora_adapter_loaded = lambda error_message="": None
    model.add_feedback_example("Prompt", "Response")
    with pytest.raises(RuntimeError):
        model.train_on_dialogue_pairs(load_from_buffer=True)


def test_train_on_existing_cover_letters_skips_if_empty(dummy_tokenizer, monkeypatch):
    monkeypatch.setattr(
        "auto_apply_bot.model_interfaces.cover_letter_generator.cover_letter_generator.load_texts_from_files",
        lambda paths: ["", "   "]
    )
    model = CoverLetterModelInterface()
    model.tokenizer = dummy_tokenizer
    result = model.train_on_existing_letters(["path"])
    assert result is None


def test_dialogue_dataset_custom_formatter(dummy_tokenizer):
    called = {}

    def custom_formatter(prompt, response):
        called["used"] = True
        return f"{prompt} -- {response}"

    data = [("Hi", "Hello")]
    dataset = DialoguePairDataset(data, dummy_tokenizer, formatter=custom_formatter)
    _ = dataset[0]
    assert called.get("used", False)


def test_generate_cover_letter_format(monkeypatch):
    captured_prompt = {}

    def fake_run(prompts, **kwargs):
        captured_prompt["value"] = prompts[0]
        return ["fake"]

    model = CoverLetterModelInterface()
    monkeypatch.setattr(model, "run_prompts", fake_run)

    model.generate_cover_letter("Cool job", ["Line1", "Line2"])
    prompt = captured_prompt["value"]

    assert "Cool job" in prompt
    assert "Relevant Experience" in prompt
    assert "Begin the cover letter" in prompt


@pytest.mark.parametrize("resume_snippets", [
    ["Did X", "Used Y"],
    ["Built ML systems", "Deployed on GCP"],
    []
])
def test_generate_cover_letter_param(monkeypatch, resume_snippets):
    model = CoverLetterModelInterface()
    monkeypatch.setattr(model, "run_prompts", lambda prompts, **kwargs: ["Fake result"])
    result = model.generate_cover_letter("Job: Do things", resume_snippets)
    assert "Fake result" in result


