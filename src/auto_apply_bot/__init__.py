from pathlib import Path


def resolve_project_source() -> Path:
    target = "auto_apply_bot"
    p = Path(__file__).resolve()
    candidates = [parent for parent in p.parents if target in parent.parts]
    if not candidates:
        raise ValueError(f'"{target}" not found in path hierarchy.')
    return candidates[-1]  # Take the highest-level one


