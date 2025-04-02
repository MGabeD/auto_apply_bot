from pathlib import Path
from functools import wraps


def resolve_highest_level_occurance_in_path(path: Path, target: str) -> Path:
    """
    Find the highest-level parent directory in the given path hierarchy that contains the target directory name.
    :param path: (Path): The path to start from (typically __file__).
    :param target: (str): The name of the directory to look for in the path hierarchy.
    :return: (Path): The highest-level matching parent path that contains the target.
    :raises: ValueError: If the target directory is not found in the path hierarchy.
    """
    candidates = [parent for parent in path.parents if target in parent.parts]
    if not candidates:
        raise ValueError(f'"{target}" not found in path hierarchy.')
    return candidates[-1]


def ensure_path_is_dir_or_create(func):
    """
    Decorator that ensures the Path returned by the function is a directory and creates it if it doesn't exist.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Path:
        path = func(*args, **kwargs)
        if not isinstance(path, Path):
            raise TypeError(f"Expected return value to be a Path, got {type(path).__name__}")

        if path.exists():
            if not path.is_dir():
                raise ValueError(f"Path '{path}' exists but is not a directory.")
        else:
            path.mkdir(parents=True, exist_ok=True)

        return path
    return wrapper