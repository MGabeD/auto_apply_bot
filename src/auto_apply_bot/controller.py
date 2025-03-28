from auto_apply_bot.model_interfaces.skill_parser import SkillParser, extract_sections
from auto_apply_bot.retrieval_interface.retrieval import LocalRagIndexer
from auto_apply_bot import resolve_project_source
from auto_apply_bot.logger import get_logger


logger = get_logger(__name__)


class Controller:
    def __init__(self, lazy_embedder: bool = False):
        logger.info("fooo")
        self.rag_engine = LocalRagIndexer(project_dir=resolve_project_source(), lazy_embedder=lazy_embedder)

        # self.rag_enginer = RAGEngine()S
        # self.alighnment_checker = AlighnmentChecker()
        # self.cover_letter_writer = CoverLetterWriter()

    def _retrieve_aligned_experiences(self, requirements: list[str]) -> list[str]:
        """
        Retrieve experiences that are aligned with the requirements
        :param requirements: List of requirements
        :return: List of aligned experiences
        """
        queries = []
        for req in requirements:
            prompt = (
                f"Retrieve specific, concrete work experiences, projects, or technical contributions "
                f"that align well with this job requirement:\n'{req}'\n"
                f"Prefer experiences from professional roles, impactful projects, or client-facing work."
            )
            queries.append(prompt)
        results = self.rag_engine.batch_query(
            query_texts=queries, 
            top_k=10, 
            similarity_threshold=0.0, 
            deduplicate=True,
        )
        return results
    
    def retrieve_and_align_experiences(self, requirements: list[str], job_description: str) -> list[str]:
        """
        Retrieve and align experiences with requirements
        :param requirements: List of requirements
        :param job_description: Raw job description text
        :return: List of aligned experiences
        """
        aligned_experiences = self._retrieve_aligned_experiences(requirements)
        prompts = []
        for skill, experiences in aligned_experiences.items():
            experience_text = "\n".join(experiences)
            prompt = (
                f"You are a highly discerning technical recruiter. "
                f"Given the following candidate experiences, extract the most relevant details for the requirement: '{skill}'. "
                f"Focus only on information that directly demonstrates proficiency or work done related to '{skill}'. "
                f"Be concise and prioritize real-world accomplishments over general statements.\n\n"
                f"Candidate Experiences:\n{experience_text}\n\n"
                f"Key points related to '{skill}':"
            )
            prompts.append(prompt)

        return prompts

    
    def generate_cover_letter(self, job_description: str, user_data_path: str, line_by_line_override: bool = False) -> str:
        """
        Full pipeline from JD and user data path to final cover letter
        :param job_description: Raw job description text
        :param user_data_path: Path to candidate's Json profile file
        :param line_by_line_override: Whether to override the line by line override
        :return: Generated cover letter
        """
        # TODO: fill this out still
        job_extracts = extract_sections(job_description)

        # 1. Extract skills from JD
        with SkillParser(device="cuda") as parser:
            skills, requirements = parser.get_job_extracts(job_reqs=job_description, user_path=user_data_path, line_by_line_override=line_by_line_override)

        # 2. Align skills with user profileA
        for req in requirements:
            logger.info(f"Requirement: {req}")
        return "foo"    
