import pytest
from io import BytesIO
from unittest import mock
from auto_apply_bot.utils.file_validation import get_mime_type, validate_file
from tests.util_tests.mocks.mock_file_validation import DummyFile


def test_get_mime_type_calls_magic(monkeypatch):
    dummy = DummyFile(b"Hello world!", name="file.txt")

    fake_magic = mock.Mock()
    fake_magic.from_buffer.return_value = "text/plain"
    monkeypatch.setattr("auto_apply_bot.utils.file_validation.mime_detector", fake_magic)

    mime = get_mime_type(dummy)
    assert mime == "text/plain"
    fake_magic.from_buffer.assert_called_once()


def test_get_mime_type_invalid_file():
    class BadFile:
        pass

    with pytest.raises(ValueError):
        get_mime_type(BadFile())


def test_validate_file_success(monkeypatch):
    dummy = DummyFile(b"Hello", name="file.txt")

    monkeypatch.setattr("auto_apply_bot.utils.file_validation.get_mime_type", lambda f: "text/plain")

    allowed = {".txt": ["text/plain"]}
    is_valid, reason = validate_file(dummy, allowed)
    assert is_valid
    assert reason is None


def test_validate_file_rejects_extension():
    dummy = DummyFile(b"hello", name="bad.exe")
    allowed = {".txt": ["text/plain"]}
    valid, reason = validate_file(dummy, allowed)
    assert not valid
    assert "Invalid file extension" in reason


def test_validate_file_exact_mimetype_match(monkeypatch):
    dummy = DummyFile(b"hello", name="note.txt")
    monkeypatch.setattr("auto_apply_bot.utils.file_validation.get_mime_type", lambda f: "text/plain")

    allowed = {".txt": ["text/plain"]}
    valid, reason = validate_file(dummy, allowed)
    assert valid is True
    assert reason is None


def test_validate_file_fuzzy_mimetype_match(monkeypatch):
    dummy = DummyFile(b"hello", name="note.txt")
    monkeypatch.setattr("auto_apply_bot.utils.file_validation.get_mime_type", lambda f: "text/plain; charset=utf-8")

    allowed = {".txt": ["text/plain"]}
    valid, reason = validate_file(dummy, allowed)
    assert valid is True
    assert reason is None


def test_validate_file_strict_mimetype_rejection(monkeypatch):
    dummy = DummyFile(b"hello", name="note.txt")
    monkeypatch.setattr("auto_apply_bot.utils.file_validation.get_mime_type", lambda f: "application/json")

    allowed = {".txt": ["text/plain"]}
    valid, reason = validate_file(dummy, allowed)
    assert not valid
    assert "Expected MIME type" in reason


def test_validate_file_exceeds_size(monkeypatch):
    dummy = DummyFile(b"x" * 2048, name="big.txt", size=2048)
    monkeypatch.setattr("auto_apply_bot.utils.file_validation.get_mime_type", lambda f: "text/plain")

    allowed = {".txt": ["text/plain"]}
    valid, reason = validate_file(dummy, allowed, max_size=1024)
    assert not valid
    assert "exceeds" in reason


def test_validate_file_single_mime_string_converted(monkeypatch):
    dummy = DummyFile(b"hello", name="note.txt")
    monkeypatch.setattr("auto_apply_bot.utils.file_validation.get_mime_type", lambda f: "text/plain")
    allowed = {".txt": "text/plain"}  # not a list
    valid, reason = validate_file(dummy, allowed)
    assert valid is True
    assert reason is None


def test_validate_file_missing_size_attribute(monkeypatch):
    dummy = DummyFile(b"hello", name="note.txt")
    del dummy.size
    monkeypatch.setattr("auto_apply_bot.utils.file_validation.get_mime_type", lambda f: "text/plain")

    allowed = {".txt": "text/plain"}
    valid, reason = validate_file(dummy, allowed, max_size=1024)
    assert not valid
    assert "File size validation failed" in reason


