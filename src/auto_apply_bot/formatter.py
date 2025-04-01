from typing import Union, List


def default_formatter(prompt: str, response: str) -> str:
    return f"### Prompt:\n{prompt.strip()}\n\n### Response:\n{response.strip()}"

def summarize_formatter(skill: str, chunks: Union[str, List[str]]) -> str:
    if isinstance(chunks, list):
        chunks = ", ".join(chunks)
    prompt = (
        f"Summarize the following experiences related to '{skill}'.\n"
        f"Focus on specific accomplishments and clearly demonstrated proficiencies.\n\n"
        f"{chunks}\n"
        f"\nSummary of relevant experience with '{skill}':"
    )
    return default_formatter(prompt, "") 

def relevance_formatter(skill: str, job_posting: str, chunks: Union[str, List[str]]) -> str:
    if isinstance(chunks, list):
        chunks = ", ".join(chunks)
    prompt = (
        f"You are a discerning hiring manager.\n"
        f"Given the job description and candidate experience, determine if this is relevant to '{skill}'.\n"
        f"Respond with 'Yes' or 'No'.\n\n"
        f"Job Description:\n{job_posting}\n\nExperience:\n{chunks}"
    )
    return default_formatter(prompt, "")