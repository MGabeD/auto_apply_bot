from typing import List, Dict, Optional, Tuple
import json
from auto_apply_bot import resolve_project_source
from transformers import BitsAndBytesConfig
from auto_apply_bot.utils.logger import get_logger
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from auto_apply_bot.model_interfaces import determine_batch_size
from auto_apply_bot.model_interfaces.base_model_interface import BaseModelInterface


logger = get_logger(__name__)


def extract_sections(job_posting: str) -> Dict[str, List[str]]:
    """
    Extracts sections from a job posting such as description, responsibilities, and requirements.
    :param job_posting: Raw job posting text
    :return: Dictionary with keys ['description', 'responsibilities', 'requirements', 'other'] and corresponding list
             of lines
    """
    sections = {
        'description': [],
        'responsibilities': [],
        'requirements': [],
        'other': []
    }

    patterns = {
        'description': r'(job description|about the role|overview)',
        'responsibilities': r'(responsibilities|duties|what you\'ll do)',
        'requirements': r'(requirements|qualifications|what we\'re looking for)'
    }

    captured_ranges = []

    for section, pattern in patterns.items():
        match = re.search(pattern, job_posting, re.IGNORECASE)
        if match:
            start = match.end()
            end = len(job_posting)
            for other_section, other_pattern in patterns.items():
                if other_section != section:
                    other_match = re.search(other_pattern, job_posting[start:], re.IGNORECASE)
                    if other_match:
                        end = start + other_match.start()
                        break
            captured_ranges.append((start, end))
            content = job_posting[start:end].strip()
            sections[section] = [line.strip() for line in content.split('\n') if len(line.strip()) >= 3]

    # Identify uncaptured parts as "other"
    uncaptured = []
    current_pos = 0
    for start, end in sorted(captured_ranges):
        if current_pos < start:
            uncaptured.append(job_posting[current_pos:start])
        current_pos = end
    if current_pos < len(job_posting):
        uncaptured.append(job_posting[current_pos:])

    other_content = "\n".join(uncaptured)
    sections['other'] = [line.strip() for line in other_content.split('\n') if len(line.strip()) >= 3]

    return sections


class SkillParser(BaseModelInterface):
    def __init__(self,
                 model_name: str = "deepseek-ai/deepseek-llm-7b-chat",
                 device: str = "cuda",
                 bnb_config: BitsAndBytesConfig = BitsAndBytesConfig( load_in_8bit=True,
                                                                      llm_int8_threshold=6.0,
                                                                      llm_int8_has_fp16_weight=False )):
        """
        Initializes the SkillParser with model parameters and device configuration.
        :param model_name: Hugging Face model name or path (default is deepseek's chat)
        :param device: Target device ('cuda' or 'cpu') (default is cuda)
        :param bnb_config: BitsAndBytesConfig object for quantization (default is for deepseek chat 7B 8bit quantized)
        """
        super().__init__(model_name, device, bnb_config)

    def cleanup(self):
        """
        Cleans up the SkillParser.
        """
        logger.warning("Cleaning up SkillParser - identical to BaseModelInterface cleanup")
        super().cleanup()

    def _split_text(self, text: str, chunk_size: int = 400) -> List[str]:
        """
        Splits a long string into smaller chunks.
        :param text: The input text to split
        :param chunk_size: The maximum size of each chunk (default = 400)
        :return: List of text chunks
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
        return splitter.split_text(text)

    def _post_process_extract_skill(self, response: str, split_string:str='\n') -> List[str]:
        """
        Processes model output to clean and extract skill items.
        :param response: Raw model response string
        :param split_string: Delimiter to split output (default = '\n')
        :return: List of cleaned and deduplicated skills (not in concept but exact string match)
        """
        cleaned = response.split("\nSkills:")[-1]
        skills = [s.strip() for s in cleaned.split(split_string) if s.strip()]
        skills = [re.sub(r'^[^a-zA-Z]+', '', s) for s in skills]
        return list(set(skills))

    def extract_skills(self, description: List[str]) -> List[str]:
        """
        Extracts relevant technical skills from job description text.
        :param description: List of lines from job description and requirements
        :return: List of extracted skills
        """
        # TODO: If I decide to do the right thing and refactor this, to use the new 
        #  run_prompts method it is everywhere where I pipe
        combined_text = " ".join(description)
        chunks = self._split_text(combined_text)
        skills = set()
        prompts = []
        for chunk in chunks:
            prompt = (
                f"As a technical recruiter, extract ONLY the specific technical skills, frameworks, tools, or methodologies a software engineer would need based on this job description. "
                f"Prioritize domain-relevant or role-specific skills (e.g., cloud platforms, ML frameworks, distributed systems). "
                f"Do NOT list general skills like 'Git' or 'Agile' unless critical. "
                f"Return a comma-separated list.\n\nJob Description:\n{chunk}\n\nSkills:"
            )
            prompts.append(prompt)
        batch_size = determine_batch_size(device_index= 0 if self.device == "cuda" else -1)
        responses = self.pipe(
            prompts,
            batch_size=batch_size,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            top_k=30,
        )
        for chunk, res in zip(chunks, responses):
            res_text = res[0]['generated_text']
            extracted = self._post_process_extract_skill(res_text)
            skills.update(extracted)
            logger.info(f"Extracted skills {extracted} from chunk {chunk}")
        return list(skills)

    def _batch_assess_requirements(self,
                                   requirements: List[str],
                                   profile: str ) -> Dict[str, Optional[int]]:
        """
        Rates how well a candidate's profile matches a specific requirement.
        :param requirements: List of job requirements
        :param profile: JSON-serialized candidate profile data
        :return: Dictionary with requirement as key and rating as value
        """
        prompts = []
        for requirement in requirements:
            prompt = (
                "You are a highly skeptical hiring manager who ONLY trusts full-time professional work experience. "
                "You do NOT trust internships, personal projects, coursework, or one-off mentions. "
                "You ONLY give a 4 or 5 if the skill is clearly present in MULTIPLE full-time professional roles.\n"
                "Use this scale:\n"
                "1 = unqualified\n"
                "2 = mentioned in personal projects/internships only\n"
                "3 = mentioned ONCE in a professional job\n"
                "4 = mentioned in TWO professional jobs\n"
                "5 = mentioned in THREE or more professional jobs.\n"
                "Respond ONLY with a single digit (1-5).\n\n"

                # First example
                "Example Requirement:\nExperience with CI/CD pipelines.\n"
                "Example Candidate Profile:\n"
                "- Built CI/CD pipeline in a capstone project.\n"
                "- Used Jenkins briefly in an internship.\n"
                "- Professional experience mentions CI/CD once in passing.\n"
                "Rating: 3\n\n"

                # Second example (harsher one)
                "Example Requirement:\nStrong SQL and NoSQL database experience.\n"
                "Example Candidate Profile:\n"
                "- Used MongoDB in personal projects.\n"
                "- Familiar with SQL from coursework.\n"
                "- Mentioned SQL during a summer internship.\n"
                "- Professional jobs do NOT mention SQL or NoSQL directly.\n"
                "Rating: 2\n\n"

                # Your real input
                f"Requirement:\n{requirement}\n\n"
                f"Candidate Profile:\n{profile}\n\n"
                "Rating:"
            )
            prompts.append(prompt)
        batch_size = determine_batch_size(device_index= 0 if self.device == 'cuda' else -1)
        logger.info(f"Running requirement assessments in batch_size={batch_size}")
        responses = self.pipe(
            prompts,
            batch_size=batch_size,
            max_new_tokens=50,
            do_sample=False,
        )
        req_ratings = {}
        for req, response in zip(requirements, responses):
            response_clean = response[0]["generated_text"]
            response_clean = response_clean.split("Rating:")[-1]

            try:
                # Look for a standalone digit 1-5
                match = int(re.search(r'(\d)', response_clean).group(1))
                if match:
                    req_ratings[req] = match
                    logger.info(f"Rating {match} for the skill or requirement of: {req}")
                else:
                    logger.warning(f"No valid rating found in model output: '{response_clean}'")
                    req_ratings[req] = None
            except Exception as e:
                logger.warning(f"Error parsing rating from: '{response_clean}' | err: {e}")
                req_ratings[req] = None
        return req_ratings

    def _asses_requirement(self, requirement: str, profile: str) -> Optional[int]:
        """
        Rates how well a candidate's profile matches a specific requirement.
        :param requirement: A single job requirement line
        :param profile: JSON-serialized candidate profile data
        :return: Rating between 1 and 5 or None if parsing fails
        """
        prompt = (
            "You are a highly skeptical hiring manager who ONLY trusts full-time professional work experience. "
            "You do NOT trust internships, personal projects, coursework, or one-off mentions. "
            "You ONLY give a 4 or 5 if the skill is clearly present in MULTIPLE full-time professional roles.\n"
            "Use this scale:\n"
            "1 = unqualified\n"
            "2 = mentioned in personal projects/internships only\n"
            "3 = mentioned ONCE in a professional job\n"
            "4 = mentioned in TWO professional jobs\n"
            "5 = mentioned in THREE or more professional jobs.\n"
            "Respond ONLY with a single digit (1-5).\n\n"

            # First example
            "Example Requirement:\nExperience with CI/CD pipelines.\n"
            "Example Candidate Profile:\n"
            "- Built CI/CD pipeline in a capstone project.\n"
            "- Used Jenkins briefly in an internship.\n"
            "- Professional experience mentions CI/CD once in passing.\n"
            "Rating: 3\n\n"

            # Second example (harsher one)
            "Example Requirement:\nStrong SQL and NoSQL database experience.\n"
            "Example Candidate Profile:\n"
            "- Used MongoDB in personal projects.\n"
            "- Familiar with SQL from coursework.\n"
            "- Mentioned SQL during a summer internship.\n"
            "- Professional jobs do NOT mention SQL or NoSQL directly.\n"
            "Rating: 2\n\n"

            # Your real input
            f"Requirement:\n{requirement}\n\n"
            f"Candidate Profile:\n{profile}\n\n"
            "Rating:"
        )

        response = self.pipe(prompt, max_new_tokens=40, do_sample=False)[0]['generated_text']
        # Extract only the "completion" part (after prompt)
        response_clean = response.split("Rating:")[-1]
        logger.warning(requirement + "\n" + response_clean)
        try:
            # Look for a standalone digit 1-5
            match = int(re.search(r'(\d)', response_clean).group(1))
            if match:
                return match
            else:
                logger.warning(f"No valid rating found in model output: '{response_clean}'")
                return None
        except Exception as e:
            logger.warning(f"Error parsing rating from: '{response_clean}' | err: {e}")
            return None

    def assess_qualifications(self, requirements: Dict[str, List[str]], profile_file: str, line_by_line_override: bool = False) -> dict:
        """
        Evaluates a candidate's profile against job requirements and returns a qualification rating.
        Performs an overall assessment first; if the rating is low or line_by_line_override is True, it falls back to rating each requirement individually.
        :param requirements: Dictionary of job requirements and/or extracted skills
        :param profile_file: Filename of the candidate's JSON profile (inside 'profile_data' directory)
        :param line_by_line_override: If True, forces detailed per-requirement scoring regardless of overall rating
        :return: Dictionary with an 'overall' rating and optionally individual ratings for each requirement
        """
        # Flatten requirements list
        flat_requirements = "\n".join([item for sublist in requirements.values() for item in sublist])

        profile_source = resolve_project_source() / "profile_data" / profile_file
        with open(profile_source, 'r', encoding='utf-8') as f:
            data = json.load(f)
        candidate_profile_data = json.dumps(data)

        # MARK: Old prompt - validate which one does better if you have time
        # prompt = (
        #     "You are a strict but fair hiring expert. You critically evaluate candidates based on actual hands-on work experience "
        #     "and give less weight to self-reported skills or extracurriculars. "
        #     "For the following job requirement and candidate profile, rate the candidate's fit based on relevant, practical experience only. "
        #     "Use a scale from 1 to 5 where 5 = overqualified, 3 = qualified, and 1 = unqualified. "
        #     "Respond ONLY with a single digit (1-5) and nothing else.\n\n"
        #     f"Requirement:\n{flat_requirements}\n\n"
        #     f"Candidate Profile:\n{candidate_profile_data}\n\n"
        #     "Rating:"
        # )
        prompt = (
            "You are a highly skeptical hiring manager who ONLY trusts full-time professional work experience. "
            "You do NOT trust internships, personal projects, coursework, or one-off mentions. "
            "You ONLY give a 4 or 5 if the skill is clearly present in MULTIPLE full-time professional roles.\n"
            "Use this scale:\n"
            "1 = unqualified\n"
            "2 = mentioned in personal projects/internships only\n"
            "3 = mentioned ONCE in a professional job\n"
            "4 = mentioned in TWO professional jobs\n"
            "5 = mentioned in THREE or more professional jobs.\n"
            "Respond ONLY with a single digit (1-5).\n\n"

            # First example
            "Example Requirement:\nExperience with CI/CD pipelines.\n"
            "Example Candidate Profile:\n"
            "- Built CI/CD pipeline in a capstone project.\n"
            "- Used Jenkins briefly in an internship.\n"
            "- Professional experience mentions CI/CD once in passing.\n"
            "Rating: 3\n\n"

            # Second example (harsher one)
            "Example Requirement:\nStrong SQL and NoSQL database experience.\n"
            "Example Candidate Profile:\n"
            "- Used MongoDB in personal projects.\n"
            "- Familiar with SQL from coursework.\n"
            "- Mentioned SQL during a summer internship.\n"
            "- Professional jobs do NOT mention SQL or NoSQL directly.\n"
            "Rating: 2\n\n"

            # Your real input
            f"Requirement:\n{flat_requirements}\n\n"
            f"Candidate Profile:\n{candidate_profile_data}\n\n"
            "Rating:"
        )
        response = self.pipe(prompt, max_new_tokens=100, do_sample=False)[0]['generated_text']
        res_cut = response.split("Rating:")[-1]
        try:
            rating = int(re.search(r'(\d)', res_cut).group(1))
        except Exception as e:
            logger.warning(f"Could not parse overall rating from: {res_cut} with err {e}")
            rating = 3

        # Fallback: break it down requirement-by-requirement if low rating
        req_ratings = dict()
        req_ratings["overall"] = rating
        if rating < 3 or line_by_line_override:
            logger.warning(f"Overall rating: {rating}/5")
            req_ratings.update(self._batch_assess_requirements(requirements=flat_requirements.split("\n"),
                                                               profile=candidate_profile_data))
            return req_ratings
        else:
            logger.info(f"Overall rating: {rating} - strong match")
        return req_ratings

    def get_job_extracts(self, job_reqs: str, user_path: str, line_by_line_override: bool = False) -> Tuple[List[str] ,dict]:
        """
        Runs full pipeline to extract skills and assess qualifications.
        :param job_reqs: Raw job posting text
        :param user_path: Profile JSON filename inside the 'profile_data' folder
        :param line_by_line_override: If True, forces detailed per-requirement scoring regardless of overall rating
        :return: Tuple of (list of enriched requirements/skills, qualification ratings dict)
        """
        data = extract_sections(job_posting=job_reqs)
        logger.info(f"Extracted sections: {list(data.keys())}")

        job_desc = data.get("description") + data.get("requirements")
        skills = self.extract_skills(description=job_desc)

        enriched_reqs = set()
        enriched_reqs.update(skills)
        enriched_reqs.update(data.get("requirements"))
        logger.info("Running qualification assessment...")
        rating = self.assess_qualifications(requirements={"requirements&skills" : list(enriched_reqs)},
                                            profile_file=user_path,
                                            line_by_line_override=line_by_line_override)
        return enriched_reqs, rating
