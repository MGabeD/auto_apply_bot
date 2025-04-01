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
    if not torch.cuda.is_available():
        if isinstance(component, dict) and component.get("device") == "cuda":
            logger.warning(f"CUDA is not available, but {name} is set to use it. Overriding device to 'cpu'.")
            component["device"] = "cpu"
        elif hasattr(component, "device") and component.device == "cuda":
            logger.warning(f"CUDA is not available, but {name} is set to use it. Overriding device to 'cpu'.")
            component.device = "cpu"
    return component


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

    def run_full_pipeline(
            self, 
            job_posting: str, 
            profile_path: str, 
            line_by_line_override: bool = False, 
            top_k_snippets: int = 5, 
            generation_kwargs: Optional[dict] = None
            ) -> str:

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

    def extract_skills(self, job_posting: str, profile_path: str, line_by_line_override: bool = False) -> tuple[list[str], dict]:
        logger.info("Extracting skills and assessing qualifications...")
        return self.skill_parser.get_job_extracts(
            job_reqs=job_posting,
            user_path=profile_path,
            line_by_line_override=line_by_line_override
        )

    def build_job_aware_queries(self, job_posting: str, skills: list[str]) -> list[str]:
        return [
            f"Given the job description: {job_posting}\n\nRetrieve experiences relevant to: '{skill}'" for skill in skills
        ]

    def query_rag(self, queries: list[str], top_k: int = 5) -> dict[str, list[dict]]:
        logger.info("Querying RAG engine with job-aware prompts...")
        return self.rag_engine.batch_query(query_texts=queries, top_k=top_k, deduplicate=True)

    def filter_relevant_chunks(self, job_posting: str, rag_results: dict[str, list[dict]]) -> dict[str, list[str]]:
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

    def summarize_grouped_chunks(self, grouped_chunks: dict[str, list[str]]) -> list[str]:
        logger.info("Summarizing grouped candidate experiences...")
        prompts = []
        for skill, chunks in grouped_chunks.items():
            combined = "\n".join(chunks)
            prompt = self.summarize_formatter(skill, combined)
            prompts.append(prompt)

        return self.cover_letter_generator.run_prompts(prompts, max_new_tokens=1024)

    def generate_cover_letter(self, job_posting: str, resume_snippets: list[str], **kwargs) -> str:
        logger.info("Generating final cover letter...")
        return self.cover_letter_generator.generate_cover_letter(
            job_description=job_posting,
            resume_snippets=resume_snippets,
            **kwargs
        )