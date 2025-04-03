from typing import Union, List
from auto_apply_bot.formatters import default_formatter


def summarize_snippets_relevant_to_skill(skill: str, chunks: Union[str, List[str]]) -> str:
    if isinstance(chunks, list):
        chunks = ", ".join(chunks)
    prompt = (
        f"Summarize the following experiences related to '{skill}'.\n"
        f"Focus on specific accomplishments and clearly demonstrated proficiencies.\n\n"
        f"{chunks}\n"
        f"\nSummary of relevant experience with '{skill}':"
    )
    return default_formatter(prompt, "")