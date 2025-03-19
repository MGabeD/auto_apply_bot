from typing import List, Dict, Optional
import json
from pathlib import Path
from auto_apply_bot import resolve_project_source
import torch
from transformers import pipeline
from auto_apply_bot.logger import get_logger
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = get_logger(__name__)


def extract_sections(job_posting: str) -> Dict[str,List[str]]:
    sections = {'description': [], 'responsibilities': [], 'requirements': []}

    # Define patterns
    patterns = {
        'description': r'(job description|about the role|overview)',
        'responsibilities': r'(responsibilities|duties|what you\'ll do)',
        'requirements': r'(requirements|qualifications|what we\'re looking for)'
    }

    # Search for each section
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
            content = job_posting[start:end].strip()
            # Split on newlines and filter out empty lines
            sections[section] = [
                line.strip() for line in content.split('\n')
                if len(line.strip()) >= 3
            ]
    return sections


class SkillParser:
    def __init__(self, model_name:str, device: str = "cuda"):
        self.model = model_name
        if device == "cuda" and not torch.cuda.is_available():
            logger.error("CUDA requested but no GPU found. Exiting Skill Parser Initialization")
            raise ValueError("CUDA requested but no GPU found. Please check your environment.")

        self.device = device
        self.pipe = pipeline("text2text-generation", model=model_name, device=0 if device == "cuda" else -1)
        logger.info(f"SkillParser initialization with model: {self.model} on device {self.device}")

    def _split_text(self, text: str, chunk_size: int = 400) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
        return splitter.split_text(text)

    def _post_process(self, response: str) -> List[str]:
        cleaned = re.sub(r'[\[\{.;]', '', response)
        return [s.strip() for s in cleaned.split(",") if s.strip()]

    def extract_skills_from_description(self, description: List[str]) -> List[str]:
        combined_text = " ".join(description)
        chunks = self._split_text(combined_text)
        skills = set()

        for chunk in chunks:
            prompt = f"Extract key technical or soft skills mentioned in the following snippet as a comma separated list of words: '{chunk}'"
            response = self.pipe(prompt, max_new_tokens=100)[0]['generated_text']
            extracted = self._post_process(response)
            skills.update(extracted)

        return list(skills)

    def extract_skills(self, job_description: str) -> List[str]:
        combined_text = " ".join(job_description)
        chunks = self._split_text(combined_text)
        skills_set = set()

        for chunk in chunks:
            prompt = f"Extract key technical or soft skills mentioned here: '{chunk}'"
            response = self.pipe(prompt, max_new_tokens=50)[0]['generated_text']
            extracted = self._post_process(response)
            skills_set.update(extracted)
        return list(skills_set)

    def asses_requirement(self, prompt: str) -> Optional[int]:
        response = self.pipe(prompt, max_new_tokens=50)[0]['generated_text']
        rating = 3
        try:
            rating = int(re.search(r'(\d)', response).group(1))
        except Exception as e:
            logger.warning(f"Could not parse model rating from: {response} with err {e}")

        if rating < 3:
            return rating
        return None

    def assess_qualifications(self, requirements: Dict[str, List[str]], profile: str) -> None:
        full_requirements = "\n".join(requirements)
        # I could just use mongodb but I don't want to scope creep super hard this is supposed to be like a two week
        # project to finish this whole base tool
        profile_source = resolve_project_source() / "profile_data" / profile
        with open(profile_source, 'r', encoding='utf-8') as f:
            data = json.load(f)
        json_str = json.dumps(data)
        prompt = ("overall on a scale of 1-5, 5 being over qualified, 1 being completely unqualified, how qualified " +
                  f"am I for a role with the requirements {full_requirements} \n my here is a full profile on my " +
                  f"experience and qualifications {json_str}")
        response = self.pipe(prompt, max_new_tokens=50)[0]['generated_text']
        rating = 3
        try:
            rating = int(re.search(r'(\d)', response).group(1))
        except Exception as e:
            logger.warning(f"Could not parse model rating from: {response} with err {e}")
        if rating >= 3:
            logger.info(f"Qualified for the role: {rating} suitable / strong fit")
        else:
            for requirement in requirements:
                pr = ("on a scale of 1-5, 5 being over qualified, 1 being completely unqualified, does this " +
                      f"requirement match my profile? {requirement} vs my profile {profile_source}")
                requirement_rating = self.asses_requirement(prompt=pr)
                if requirement_rating is not None:
                    logger.warning(f"May be under-qualified (rating: {requirement_rating} for the requirement: {requirement}")
