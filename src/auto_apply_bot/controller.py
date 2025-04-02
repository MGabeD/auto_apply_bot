from auto_apply_bot.model_interfaces.skill_parser import SkillParser, extract_sections
from auto_apply_bot.model_interfaces.cover_letter_generator.cover_letter_generator import CoverLetterModelInterface
from auto_apply_bot.retrieval_interface.retrieval import LocalRagIndexer
from auto_apply_bot import resolve_project_source
from auto_apply_bot.logger import get_logger
from typing import Optional, Union, List, Callable, Dict
import torch
from auto_apply_bot.formatter import summarize_formatter, relevance_formatter


logger = get_logger(__name__)


RelevanceFormatter = Callable[[str, str, Union[str, List[str]]], str]
SummarizeFormatter = Callable[[str, Union[str, List[str]]], str]


def _safe_default_override(component: Union[dict, object], name: str) -> Union[dict, object]:
    """
    If CUDA is not available, override the device to 'cpu' if it's set to 'cuda' in the constructor kwargs.
    """
    if not torch.cuda.is_available():
        if isinstance(component, dict) and component.get("device") == "cuda":
            logger.warning(f"CUDA is not available, but {name} is set to use it. Overriding device to 'cpu'.")
            component["device"] = "cpu"
        elif hasattr(component, "device") and component.device == "cuda":
            logger.warning(f"CUDA is not available, but {name} is set to use it. Overriding device to 'cpu'.")
            component.device = "cpu"
    return component


def ensure_pipe_loaded(module_attr: str):
    """
    Decorator to auto-enter and exit the context manager of a model interface
    if its `.pipe` is not initialized.
    """
    def decorator(fn):
        def wrapper(self, *args, **kwargs):
            module = getattr(self, module_attr)
            if getattr(module, "pipe", None) is not None:
                return fn(self, *args, **kwargs)
            with module:
                return fn(self, *args, **kwargs)
        return wrapper
    return decorator


class Controller:
    def __init__(
        self,
        skill_parser: Union[dict, SkillParser] = {"device": "cuda"},
        rag_engine: Union[dict, LocalRagIndexer] = {"project_dir": resolve_project_source(), "lazy_embedder": True},
        cover_letter_generator: Union[dict, CoverLetterModelInterface] = {"device": "cuda"},
        relevance_formatter: RelevanceFormatter = relevance_formatter,
        summarize_formatter: SummarizeFormatter = summarize_formatter,
    ):
        """
        Initializes the Controller with optionally pre-instantiated modules or constructor arguments as dicts.
        :param skill_parser: Either a SkillParser instance or a dict of constructor kwargs.
        :param rag_engine: Either a LocalRagIndexer instance or a dict of constructor kwargs.
        :param cover_letter_generator: Either a CoverLetterModelInterface instance or a dict of constructor kwargs.
        :param relevance_formatter: A function that formats a relevance check prompt.
        :param summarize_formatter: A function that formats a summarize prompt.
        """
        def _resolve_component(value, default_cls):
            if isinstance(value, dict):
                return default_cls(**value)
            elif value is not None:
                return value
            return default_cls()

        self.skill_parser = _safe_default_override(_resolve_component(skill_parser, SkillParser), "Skill parser")
        self.rag_engine = _safe_default_override(_resolve_component(rag_engine, lambda: LocalRagIndexer(project_dir=resolve_project_source())), "RAG engine")
        self.cover_letter_generator = _safe_default_override(_resolve_component(cover_letter_generator, CoverLetterModelInterface), "Cover letter generator")
        self.relevance_formatter: RelevanceFormatter = relevance_formatter
        self.summarize_formatter: SummarizeFormatter = summarize_formatter
    
        self.last_run_data: Dict[str, object] = {}

        logger.info("Controller initialized with components:")
        logger.info(f" - SkillParser: {type(self.skill_parser).__name__}")
        logger.info(f" - LocalRagIndexer: {type(self.rag_engine).__name__}")
        logger.info(f" - CoverLetterModelInterface: {type(self.cover_letter_generator).__name__}")
        logger.info(f" - RelevanceFormatter: {type(self.relevance_formatter).__name__}")
        logger.info(f" - SummarizeFormatter: {type(self.summarize_formatter).__name__}")

    def cleanup(self):
        """
        Cleans up the Controller.
        """
        logger.warning("Cleaning up Controller's via delegating to sub-elements' cleanup")
        self.skill_parser.cleanup()
        self.rag_engine.cleanup()
        self.cover_letter_generator.cleanup()

    def run_full_pipeline(
            self, 
            job_posting: str, 
            profile_path: str, 
            line_by_line_override: bool = False, 
            top_k_snippets: int = 5, 
            generation_kwargs: Optional[dict] = None
            ) -> str:
        """
        Runs the full pipeline for generating a cover letter.
        :param job_posting: The job posting to generate a cover letter for.
        :param profile_path: The path to the user's profile.
        :param line_by_line_override: Whether to use line-by-line processing.
        :param top_k_snippets: The number of snippets to retrieve from RAG.
        :param generation_kwargs: Additional kwargs for the cover letter generator.
        """
        with self.skill_parser as parser:
            skills, qualification_scores = parser.get_job_extracts(
                job_reqs=job_posting,
                user_path=profile_path,
                line_by_line_override=line_by_line_override
            )

        job_aware_queries = self.build_job_aware_queries(job_posting, skills)

        rag_results = self.query_rag(job_aware_queries, top_k=top_k_snippets)

        with self.skill_parser as parser:
            filtered_chunks = self.filter_relevant_chunks(job_posting, rag_results)

        with self.cover_letter_generator as generator:
            summarized_experiences = self.summarize_grouped_chunks(filtered_chunks)

            cover_letter = generator.generate_cover_letter(
                job_description=job_posting,
                resume_snippets=summarized_experiences,
                **(generation_kwargs or {})
            )

        self.last_run_data = {
            "skills": skills,
            "qualification_scores": qualification_scores,
            "job_aware_queries": job_aware_queries,
            "rag_results": rag_results,
            "filtered_chunks": filtered_chunks,
            "summarized_experiences": summarized_experiences,
            "cover_letter": cover_letter
        }
        logger.info(f"Logging last run data into file", extra={"data": self.last_run_data})
        return cover_letter

    @ensure_pipe_loaded("skill_parser")
    def extract_skills(self, job_posting: str, profile_path: str, line_by_line_override: bool = False) -> tuple[list[str], dict]:
        """
        Extracts skills and assesses qualifications from the job posting and profile.
        :param job_posting: The job posting to extract skills from.
        :param profile_path: The path to the user's profile.
        :param line_by_line_override: Whether to use line-by-line processing.
        """
        logger.info("Extracting skills and assessing qualifications...")
        return self.skill_parser.get_job_extracts(
            job_reqs=job_posting,
            user_path=profile_path,
            line_by_line_override=line_by_line_override
        )

    def build_job_aware_queries(self, job_posting: str, skills: list[str]) -> list[str]:
        """
        Builds job-aware queries for the RAG engine.
        :param job_posting: The job posting to build queries from.
        :param skills: The skills to build queries for.
        """
        return [
            f"Given the job description: {job_posting}\n\nRetrieve experiences relevant to: '{skill}'" for skill in skills
        ]

    def query_rag(self, queries: list[str], top_k: int = 5) -> dict[str, list[dict]]:
        """
        Queries the RAG engine with job-aware prompts.
        :param queries: The queries to query the RAG engine with.
        :param top_k: The number of results to return.
        """
        logger.info("Querying RAG engine with job-aware prompts...")
        return self.rag_engine.batch_query(query_texts=queries, top_k=top_k, deduplicate=True)

    @ensure_pipe_loaded("skill_parser")
    def filter_relevant_chunks(self, job_posting: str, rag_results: dict[str, list[dict]]) -> dict[str, list[str]]:
        """
        Filters the RAG results by LLM relevance check.
        :param job_posting: The job posting to filter the RAG results by.
        :param rag_results: The RAG results to filter.
        """
        logger.info("Filtering RAG results by LLM relevance check...")

        prompts = []
        skill_lookup = []
        match_lookup = []

        for skill, matches in rag_results.items():
            for match in matches:
                prompt = self.relevance_formatter(skill, job_posting, match["text"])
                prompts.append(prompt)
                skill_lookup.append(skill)
                match_lookup.append(match)

        responses = self.skill_parser.run_prompts(prompts)
        relevant_chunks = {skill: [] for skill in rag_results}

        for skill, match, response in zip(skill_lookup, match_lookup, responses):
            if response.lower().strip().startswith("yes"):
                relevant_chunks[skill].append(match["text"].strip())

        if not any(relevant_chunks.values()):
            logger.warning("No relevant chunks found after LLM filtering.")

        return relevant_chunks

    @ensure_pipe_loaded("cover_letter_generator")
    def summarize_grouped_chunks(self, grouped_chunks: dict[str, list[str]]) -> list[str]:
        """
        Summarizes the grouped candidate experiences.
        :param grouped_chunks: The grouped candidate experiences to summarize.
        """
        logger.info("Summarizing grouped candidate experiences...")
        prompts = []
        for skill, chunks in grouped_chunks.items():
            combined = "\n".join(chunks)
            prompt = self.summarize_formatter(skill, combined)
            prompts.append(prompt)

        return self.cover_letter_generator.run_prompts(prompts, max_new_tokens=1024)

    @ensure_pipe_loaded("cover_letter_generator")
    def generate_cover_letter(self, job_posting: str, resume_snippets: list[str], **kwargs) -> str:
        """
        Generates a cover letter.
        :param job_posting: The job posting to generate a cover letter for.
        :param resume_snippets: The snippets to include in the cover letter.
        :param kwargs: Additional kwargs for the cover letter generator.
        """
        logger.info("Generating final cover letter...")
        return self.cover_letter_generator.generate_cover_letter(
            job_description=job_posting,
            resume_snippets=resume_snippets,
            **kwargs
        )

