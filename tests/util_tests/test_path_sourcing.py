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
    assert isinstance(result2, Path)


def test_resolve_highest_level_occurance_failure():
    test_path = Path("/a/b/c/d")
    with pytest.raises(ValueError):
        resolve_highest_level_occurance_in_path(test_path, "target")


def test_ensure_path_is_dir_or_create_existing_directory_succeeds(tmp_path):
    existing_dir = tmp_path / "existing"
    existing_dir.mkdir()

    @ensure_path_is_dir_or_create
    def return_existing():
        return existing_dir

    result = return_existing()
    assert result == existing_dir
    assert result.exists() and result.is_dir()


def test_ensure_path_is_dir_or_create_on_path_that_is_file_succeeds(tmp_path):
    non_dir_path = tmp_path / "non_dir"
    non_dir_path.touch()

    @ensure_path_is_dir_or_create
    def return_non_dir():
        return non_dir_path
    
    with pytest.raises(ValueError):
        return_non_dir()
    

def test_ensure_path_is_dir_or_create_creates_directory_if_it_does_not_exist(tmp_path):
    non_existent_dir = tmp_path / "non_existent"

    @ensure_path_is_dir_or_create
    def return_non_existent():
        return non_existent_dir
    
    result = return_non_existent()
    assert result == non_existent_dir
    assert result.exists() and result.is_dir()


def test_ensure_path_error_on_non_path_object(tmp_path):
    @ensure_path_is_dir_or_create
    def return_non_path():
        return 1

    with pytest.raises(TypeError):
        return_non_path()



