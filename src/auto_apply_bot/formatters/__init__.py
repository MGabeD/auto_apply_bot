from typing import Protocol, Union, List


def default_formatter(prompt: str, response: str) -> str:
    return f"### Prompt:\n{prompt.strip()}\n\n### Response:\n{response.strip()}"


class RelevanceFormatter(Protocol):
    def __call__(self, skill: str, job_posting: str, chunks: Union[str, List[str]]) -> str: ...


class SummarizeFormatter(Protocol):
    def __call__(self, skill: str, chunks: Union[str, List[str]]) -> str: ...

