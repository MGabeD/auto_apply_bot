from typing import List, Dict, Optional, Tuple
import json
from auto_apply_bot import resolve_project_source
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from auto_apply_bot.logger import get_logger
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from accelerate import infer_auto_device_map, init_empty_weights


logger = get_logger(__name__)


def log_free_memory(device_index: int = 0) -> float:
    free_mem, total_mem = torch.cuda.mem_get_info(device_index)
    free_mem_gb = free_mem / 1e9
    logger.info(f"GPU free memory: {free_mem_gb:.2f} GB")
    return free_mem_gb


def extract_sections(job_posting: str) -> Dict[str, List[str]]:
    sections = {
        'description': [],
        'responsibilities': [],
        'requirements': [],
        'other': []
    }

    # Define patterns
    patterns = {
        'description': r'(job description|about the role|overview)',
        'responsibilities': r'(responsibilities|duties|what you\'ll do)',
        'requirements': r'(requirements|qualifications|what we\'re looking for)'
    }

    # Keep track of all ranges we capture
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


class SkillParser:
    def __init__(self,
                 model_name: str = "deepseek-ai/deepseek-llm-7b-chat",
                 device: str = "cuda",
                 bnb_config: BitsAndBytesConfig = BitsAndBytesConfig( load_in_8bit=True,
                                                                      llm_int8_threshold=6.0,
                                                                      llm_int8_has_fp16_weight=False )):
        self.model_name = model_name
        self.device = device
        self.pipe = None
        self.bnb_config = bnb_config

    def __enter__(self):
        torch.cuda.empty_cache()
        log_free_memory()
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        try:
            logger.info("Trying full GPU load...")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map={"": 0},
                quantization_config=self.bnb_config,
                trust_remote_code=True
            )
            logger.info("Loaded fully on GPU.")
        except Exception as e:
            logger.warning(f"GPU memory insufficient, falling back to GPU+CPU split. Err: {e}")
            with init_empty_weights():
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=self.bnb_config,
                    trust_remote_code=True
                )

            device_map = infer_auto_device_map(
                model,
                max_memory={0: "16GiB", "cpu": "64GiB"},
                no_split_module_classes=["DecoderLayer"]
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=device_map,
                quantization_config=self.bnb_config,
                trust_remote_code=True
            )
            logger.warning(f"Loaded model with CPU fallback. Device map: {device_map}")

        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )
        log_free_memory()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pipe = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("Pipeline and GPU memory successfully released.")

    def _split_text(self, text: str, chunk_size: int = 400) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
        return splitter.split_text(text)

    def _post_process(self, response: str, split_string:str='\n') -> List[str]:
        # response = re.sub(r'^(skills|answer|result|output)[:\s-]*', '', response, flags=re.IGNORECASE)
        # cleaned = re.sub(r'[^a-zA-Z0-9+/#&,\s-]', '', response)
        cleaned = response.split("\nSkills:")[-1]
        skills = [s.strip() for s in cleaned.split(split_string) if s.strip()]
        skills = [re.sub(r'^[^a-zA-Z]+', '', s) for s in skills]
        return list(set(skills))

    def extract_skills(self, description: List[str]) -> List[str]:
        combined_text = " ".join(description)
        chunks = self._split_text(combined_text)
        skills = set()

        for chunk in chunks:
            prompt = (
                f"As a technical recruiter, extract ONLY the specific technical skills, frameworks, tools, or methodologies a software engineer would need based on this job description. "
                f"Prioritize domain-relevant or role-specific skills (e.g., cloud platforms, ML frameworks, distributed systems). "
                f"Do NOT list general skills like 'Git' or 'Agile' unless critical. "
                f"Return a comma-separated list.\n\nJob Description:\n{chunk}\n\nSkills:"
            )
            response = self.pipe(
                prompt,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                top_k=30,
            )
            res_text= response[0]['generated_text']
            extracted = self._post_process(res_text)
            skills.update(extracted)
        logger.info(f"Extracted skills {skills}")
        return list(skills)

    def _asses_requirement(self, requirement: str, profile: str) -> Optional[int]:
        prompt = (
            "Rate from 1 to 5 how well this profile matches the following job requirement. "
            "Respond ONLY with a single digit (1-5). 5 = overqualified, 1 = unqualified.\n\n"
            f"Requirement: {requirement}\n\nProfile: {profile}\n\nAnswer:"
        )
        response = self.pipe(prompt, max_new_tokens=10, do_sample=False)[0]['generated_text']

        # Extract only the "completion" part (after prompt)
        response_clean = response.split("Answer:")[-1].strip()

        try:
            # Look for a standalone digit 1-5
            match = re.search(r'\b([1-5])\b', response_clean)
            if match:
                return int(match.group(1))
            else:
                logger.warning(f"No valid rating found in model output: '{response_clean}'")
                return None
        except Exception as e:
            logger.warning(f"Error parsing rating from: '{response_clean}' | err: {e}")
            return None

    def assess_qualifications(self, requirements: Dict[str, List[str]], profile_file: str) -> dict:
        # Flatten requirements list
        flat_requirements = "\n".join([item for sublist in requirements.values() for item in sublist])

        profile_source = resolve_project_source() / "profile_data" / profile_file
        with open(profile_source, 'r', encoding='utf-8') as f:
            data = json.load(f)
        json_str = json.dumps(data)

        # MAIN prompt
        prompt = (
            "You are a hiring expert. On a scale of 1 to 5 (5 = over qualified, 1 = unqualified, 3 = qualified), "
            "how well does this candidate's profile match the following role requirements? "
            "Respond ONLY with a single digit (1-5).\n\n"
            f"Requirements:\n{flat_requirements}\n\nProfile:\n{json_str}\n\n Qualification Rating:"
        )
        response = self.pipe(prompt, max_new_tokens=100, do_sample=False)[0]['generated_text']
        res_cut = response.split("Rating:")
        res_cut = res_cut[-1]
        try:
            rating = int(re.search(r'(\d)', res_cut).group(1))
        except Exception as e:
            logger.warning(f"Could not parse overall rating from: {res_cut} with err {e}")
            rating = 3

        # Fallback: break it down requirement-by-requirement if low rating
        req_ratings = dict()
        req_ratings["overall"] = rating
        if rating < 3:
            for req in flat_requirements.split("\n"):
                req_rating = self._asses_requirement(requirement=req, profile=json_str)
                req_ratings[req] = req_rating
                if req_rating is not None and req_rating < 3:
                    logger.warning(f"Potential weak match: {req_rating}/5 for requirement: {req}")
                else:
                    logger.info(f"Potential match in skills! {req_rating}/5 for requirement: {req}")
            return req_ratings
        else:
            logger.info(f"Overall rating: {rating} - strong match")
        return req_ratings


    def get_job_extracts(self, job_reqs: str, user_path: str) -> Tuple[List[str] ,dict]:
        data = extract_sections(job_posting=job_reqs)
        logger.info(f"Extracted sections: {list(data.keys())}")

        job_desc = data.get("description") + data.get("requirements")
        skills = self.extract_skills(description=job_desc)

        enriched_reqs = set()
        enriched_reqs.update(skills)
        enriched_reqs.update(data.get("requirements"))
        logger.info("Running qualification assessment...")
        rating = self.assess_qualifications(requirements={"requirements&skills" : list(enriched_reqs)}, profile_file=user_path)

        return enriched_reqs, rating
