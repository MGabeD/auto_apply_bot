from typing import Union, List
from auto_apply_bot.formatters import default_formatter


def check_relevant_skills_as_hiring_manager(skill: str, job_posting: str, chunks: Union[str, List[str]]) -> str:
    if isinstance(chunks, list):
        chunks = ", ".join(chunks)
    prompt = (
        f"You are a discerning hiring manager.\n"
        f"Given the job description and candidate experience, determine if this is relevant to '{skill}'.\n"
        f"Respond with 'Yes' or 'No'.\n\n"
        f"Job Description:\n{job_posting}\n\nExperience:\n{chunks}"
    )
    return default_formatter(prompt, "")