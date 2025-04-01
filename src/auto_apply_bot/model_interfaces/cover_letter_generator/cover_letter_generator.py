from auto_apply_bot.model_interfaces.lora_model_interface import LoraModelInterface, LoraTrainingDataset
from auto_apply_bot.loader import load_texts_from_files
from auto_apply_bot.logger import get_logger
from transformers import PreTrainedTokenizer
from typing import List, Optional
from pathlib import Path
import uuid


logger = get_logger(__name__)


class CoverLetterModelInterface(LoraModelInterface):
    def generate_cover_letter(self, job_description: str, resume_snippets: List[str], **kwargs) -> str:
        prompt = (
            "You are a helpful assistant tasked with writing a personalized cover letter.\n"
            "Use the following job description and resume snippets to create a clear, compelling narrative.\n"
            f"Job Description:\n{job_description}\n"
            f"Relevant Experience:\n{chr(10).join(resume_snippets)}\n"
            "Begin the cover letter with a warm introduction and end with a confident closing."
        )
        return self.run_prompts([prompt], **kwargs)[0]

    def assess_experience_against_posting(self, job_description: str, experience_snippets: List[str], **kwargs) -> str:
        prompt = (
            "Evaluate how well the following experience matches the job description.\n"
            f"Job Description:\n{job_description}\n"
            f"Candidate Experience:\n{chr(10).join(experience_snippets)}\n"
            "Provide a strengths/weaknesses analysis and a 1-10 match rating."
        )
        return self.run_prompts([prompt], **kwargs)[0]

    def train_on_existing_letters(self, letter_paths: List[str], output_subdir_override: Optional[str] = None):
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer must be loaded before training. Use within a context manager.")
        dataset = LoraTrainingDataset(letter_paths, self.tokenizer)
        return self.fine_tune(train_dataset=dataset, output_subdir_override=output_subdir_override)

    def train_on_conversations(self, dialogue_pairs: List[tuple[str, str]], output_subdir_override: Optional[str] = None):
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer must be loaded before training. Use within a context manager.")
        dataset = DialoguePairDataset(dialogue_pairs, self.tokenizer)
        return self.fine_tune(train_dataset=dataset, output_subdir_override=output_subdir_override)

    def interactive_training_mode(self, session_id: Optional[str] = None):
        session_id = session_id or str(uuid.uuid4())
        logger.info(f"Interactive RLHF training session started. Session ID: {session_id}\n")
        collected_pairs = []
        try:
            while True:
                user_input = input("[You] > ").strip()
                if user_input.lower() in {"exit", "quit"}:
                    break
                response = self.run_prompts([user_input])[0]
                logger.info(f"[Model] {response}")
                accept = input("Accept this response for training? (y/n): ").strip().lower()
                if accept == "y":
                    collected_pairs.append((user_input, response))
        except KeyboardInterrupt:
            logger.error("Session interrupted.")

        if collected_pairs:
            logger.info(f"Collected {len(collected_pairs)} dialogue pairs. Starting training...")
            return self.train_on_conversations(collected_pairs, output_subdir_override=f"interactive_{session_id}")
        else:
            logger.error("No examples collected. Exiting without training.")


class DialoguePairDataset(LoraTrainingDataset):
    def __init__(self, dialogue_pairs: List[tuple[str, str]], tokenizer: PreTrainedTokenizer,  max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs = dialogue_pairs

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        prompt, response = self.pairs[idx]
        text = f"### Prompt:\n{prompt.strip()}\n\n### Response:\n{response.strip()}"
        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

