from pathlib import Path
from auto_apply_bot.utils.path_sourcing import resolve_highest_level_occurance_in_path, ensure_path_is_dir_or_create


PROJECT_NAME = "auto_apply_bot"


def resolve_project_source() -> Path:
    """
    Resolves the project source directory based on the file path of the current module.
    """
    return resolve_highest_level_occurance_in_path(Path(__file__).resolve(), PROJECT_NAME)


@ensure_path_is_dir_or_create
def resolve_component_dirs_path(component_name: str) -> Path:
    """
    Resolves the path to the directory containing the component's subdirectories.
    :param component_name: (str): The name of the component.
    :return: (Path): The path to the component's subdirectories.
    """
    return resolve_project_source() / component_name
