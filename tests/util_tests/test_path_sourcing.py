import pytest
from auto_apply_bot.utils.path_sourcing import resolve_highest_level_occurance_in_path, ensure_path_is_dir_or_create
from pathlib import Path


def test_resolve_highest_level_occurance_success():
    test_path = Path("/a/b/target/c/d")
    result = resolve_highest_level_occurance_in_path(test_path, "target")
    assert result == Path("/a/b/target")
    test_path2 = Path("/a/b/target/target/c/d")
    result2 = resolve_highest_level_occurance_in_path(test_path2, "target")
    assert result2 == Path("/a/b/target")


def test_resolve_highest_level_occurance_failure():
    test_path = Path("/a/b/c/d")
    with pytest.raises(ValueError):
        resolve_highest_level_occurance_in_path(test_path, "target")
