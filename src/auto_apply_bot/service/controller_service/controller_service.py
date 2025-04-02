from auto_apply_bot.controller import Controller
from auto_apply_bot.model_interfaces.skill_parser import SkillParser
from auto_apply_bot.retrieval_interface.retrieval import LocalRagIndexer
from auto_apply_bot.model_interfaces.cover_letter_generator.cover_letter_generator import CoverLetterModelInterface
from auto_apply_bot.formatter import RelevanceFormatter, SummarizeFormatter
from typing import Optional, Union
from threading import Lock

_controller = None
_controller_lock = Lock()


def get_controller(
    skill_parser: Optional[Union[dict, SkillParser]] = None,
    rag_engine: Optional[Union[dict, LocalRagIndexer]] = None,
    cover_letter_generator: Optional[Union[dict, CoverLetterModelInterface]] = None,
    relevance_formatter: Optional[RelevanceFormatter] = None,
    summarize_formatter: Optional[SummarizeFormatter] = None,
    force_reload: bool = False,
) -> Controller:
    global _controller
    with _controller_lock:
        if _controller is None or force_reload:
            _controller = Controller(
                skill_parser=skill_parser,
                rag_engine=rag_engine,
                cover_letter_generator=cover_letter_generator,
                relevance_formatter=relevance_formatter,
                summarize_formatter=summarize_formatter,
            )
        return _controller
