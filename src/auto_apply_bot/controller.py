from auto_apply_bot.model_interfaces.skill_parser import SkillParser, extract_sections
from auto_apply_bot.model_interfaces.cover_letter_generator.cover_letter_generator import CoverLetterModelInterface
from auto_apply_bot.retrieval_interface.retrieval import LocalRagIndexer
from auto_apply_bot import resolve_project_source
from auto_apply_bot.logger import get_logger
from typing import Optional, Union
import torch


logger = get_logger(__name__)


class Controller:
    def __init__(
        self,
        skill_parser: Union[dict, SkillParser] = {"device": "cuda" if torch.cuda.is_available() else "cpu"},
        rag_engine: Union[dict, LocalRagIndexer] = {"project_dir": resolve_project_source(), "lazy_embedder": True},
        cover_letter_generator: Union[dict, CoverLetterModelInterface] = {"device": "cuda" if torch.cuda.is_available() else "cpu"},
    ):
        """
        Initializes the Controller with optionally pre-instantiated modules or constructor arguments as dicts.
        :param skill_parser: Either a SkillParser instance or a dict of constructor kwargs.
        :param rag_engine: Either a LocalRagIndexer instance or a dict of constructor kwargs.
        :param cover_letter_generator: Either a CoverLetterModelInterface instance or a dict of constructor kwargs.
        """

        def _build_component(value, default_cls):
            if isinstance(value, dict):
                return default_cls(**value)
            elif value is not None:
                return value
            return default_cls()

        self.skill_parser = _build_component(skill_parser, SkillParser)
        self.rag_engine = _build_component(rag_engine, lambda: LocalRagIndexer(project_dir=resolve_project_source()))
        self.cover_letter_generator = _build_component(cover_letter_generator, CoverLetterModelInterface)

        logger.info("Controller initialized with components:")
        logger.info(f" - SkillParser: {type(self.skill_parser).__name__}")
        logger.info(f" - LocalRagIndexer: {type(self.rag_engine).__name__}")
        logger.info(f" - CoverLetterModelInterface: {type(self.cover_letter_generator).__name__}")

    def run_full_pipeline(
        self,
        job_posting: str,
        profile_path: str,
        line_by_line_override: bool = False,
        top_k_snippets: int = 5,
        generation_kwargs: Optional[dict] = None
    ) -> str:
        skills, qualification_scores = self.extract_skills(job_posting, profile_path, line_by_line_override)
        job_aware_queries = self.build_job_aware_queries(job_posting, skills)
        rag_results = self.query_rag(job_aware_queries, top_k=top_k_snippets)
        filtered_chunks = self.filter_relevant_chunks(job_posting, rag_results)
        summarized_experiences = self.summarize_grouped_chunks(filtered_chunks)
        return self.generate_cover_letter(job_posting, summarized_experiences, **(generation_kwargs or {}))

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

        for skill, matches in rag_results.items():
            for match in matches:
                prompt = (
                    f"You are a discerning hiring manager.\n"
                    f"Given the job description and candidate experience, determine if this is relevant to '{skill}'.\n"
                    f"Respond with 'Yes' or 'No'.\n\n"
                    f"Job Description:\n{job_posting}\n\nExperience:\n{match['text']}"
                )
                prompts.append(prompt)
                skill_lookup.append(skill)

        responses = self.skill_parser.run_prompts(prompts)
        relevant_chunks = {skill: [] for skill in rag_results}

        for skill, match, response in zip(skill_lookup, prompts, responses):
            if response.lower().strip().startswith("yes"):
                relevant_chunks[skill].append(match.split("Experience:\n")[-1].strip())

        return relevant_chunks

    def summarize_grouped_chunks(self, grouped_chunks: dict[str, list[str]]) -> list[str]:
        logger.info("Summarizing grouped candidate experiences...")
        prompts = []
        for skill, chunks in grouped_chunks.items():
            combined = "\n".join(chunks)
            prompt = (
                f"Summarize the following experiences related to '{skill}'.\n"
                f"Focus on specific accomplishments and clearly demonstrated proficiencies.\n\n"
                f"{combined}\n"
                f"\nSummary of relevant experience with '{skill}':"
            )
            prompts.append(prompt)

        return self.cover_letter_generator.run_prompts(prompts, max_new_tokens=250)

    def generate_cover_letter(self, job_posting: str, resume_snippets: list[str], **kwargs) -> str:
        logger.info("Generating final cover letter...")
        return self.cover_letter_generator.generate_cover_letter(
            job_description=job_posting,
            resume_snippets=resume_snippets,
            **kwargs
        )