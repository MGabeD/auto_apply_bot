import pytest
from pathlib import Path
from auto_apply_bot.utils.loader import load_documents, load_texts_from_files
from langchain_core.documents import Document
from tests.mocks.mock_loader import DummyLoader


def test_load_documents_success(monkeypatch, tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("dummy content")

    loader_map = {".txt": DummyLoader}
    docs = load_documents([test_file], loader_map_override=loader_map)

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "This is a dummy document."
    assert docs[0].metadata["source"] == str(test_file)


def test_load_documents_unsupported_extension(tmp_path):
    test_file = tmp_path / "file.unsupported"
    test_file.write_text("invalid")

    with pytest.raises(ValueError, match="Unsupported file type: .unsupported"):
        load_documents([test_file])


def test_load_texts_from_files(monkeypatch, tmp_path):
    test_file = tmp_path / "sample.txt"
    test_file.write_text("whatever")

    loader_map = {".txt": DummyLoader}
    monkeypatch.setattr("auto_apply_bot.utils.loader.LOADER_MAP", loader_map)

    texts = load_texts_from_files([test_file])
    assert texts == ["This is a dummy document."]
