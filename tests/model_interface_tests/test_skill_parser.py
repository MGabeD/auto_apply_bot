import pytest
from auto_apply_bot.model_interfaces.skill_parser import determine_batch_size, extract_sections


def test_determine_batch_size(monkeypatch):
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda device_idx: (12e9, 16e9))
    assert determine_batch_size() == 8

    monkeypatch.setattr("torch.cuda.mem_get_info", lambda device_idx: (7e9, 16e9))
    assert determine_batch_size() == 4

    monkeypatch.setattr("torch.cuda.mem_get_info", lambda device_idx: (3e9, 16e9))
    assert determine_batch_size() == 2

    monkeypatch.setattr("torch.cuda.mem_get_info", lambda device_idx: (1e9, 16e9))
    assert determine_batch_size() == 1


def test_extract_sections_basic():
    sample = """
    Job Description
    This is a sample job.

    Responsibilities
    - Build APIs
    - Work with databases

    Requirements
    - Python
    - SQL
    """
    sections = extract_sections(sample)
    assert "description" in sections and sections["description"]
    assert "responsibilities" in sections and sections["responsibilities"]
    assert "requirements" in sections and sections["requirements"]
    assert "other" in sections


def test_extract_sections_missing_headers():
    sample = "This posting has no headers at all."
    sections = extract_sections(sample)
    assert sections["other"]


def test_split_text(mock_parser):
    chunks = mock_parser._split_text("A long string " * 50)
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert len(chunks) > 0


def test_post_process_extract_skill(mock_parser):
    response = "\nSkills:\nPython\nDjango\nPython\n"
    skills = mock_parser._post_process_extract_skill(response)
    assert "Python" in skills
    assert "Django" in skills


def test_extract_skills(mock_parser):
    mock_parser.pipe.return_value = [[{'generated_text': '\nSkills:\nPython\nFlask\n'}]]
    skills = mock_parser.extract_skills(["We need Python and Flask developers."])
    assert "Python" in skills and "Flask" in skills


def test_batch_assess_requirements(mock_parser):
    mock_parser.pipe.return_value = [[{'generated_text': 'Rating: 3'}]] * 2
    results = mock_parser._batch_assess_requirements(["SQL", "AWS"], profile="dummy profile")
    assert results["SQL"] == 3 and results["AWS"] == 3


def test_assess_qualifications(mock_parser, monkeypatch, tmp_path):
    monkeypatch.setattr("auto_apply_bot.model_interfaces.skill_parser.SkillParser._batch_assess_requirements", lambda *args, **kwargs: {"SQL": 2})
    mock_parser.pipe.return_value = [{'generated_text': 'Rating: 2'}]
    monkeypatch.setattr("auto_apply_bot.model_interfaces.skill_parser.resolve_project_source", lambda: tmp_path)

    profile_dir = tmp_path / "profile_data"
    profile_dir.mkdir()
    profile_file = profile_dir / "dummy_profile.json"
    profile_file.write_text('{"summary": "Experienced developer"}')
    results = mock_parser.assess_qualifications({"requirements": ["SQL"]}, "dummy_profile.json")
    assert results["overall"] == 2
    assert results["SQL"] == 2

