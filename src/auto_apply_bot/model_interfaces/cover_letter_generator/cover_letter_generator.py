from auto_apply_bot.model_interfaces.lora_model_interface import LoraModelInterface, LoraTrainingDataset
from auto_apply_bot.loader import load_texts_from_files
from auto_apply_bot.logger import get_logger
from transformers import PreTrainedTokenizer, BitsAndBytesConfig
from typing import List, Optional
from pathlib import Path
import uuid


logger = get_logger(__name__)


class CoverLetterModelInterface(LoraModelInterface):
    
    def __init__(self,
                 model_name: str = "deepseek-ai/deepseek-llm-7b-chat",
                 device: str = "cuda",
                 lora_weights_dir: Optional[str] = None,
                 lora_weights_file_override: Optional[str] = None,
                 bnb_config: BitsAndBytesConfig = BitsAndBytesConfig(load_in_8bit=True,
                                                                     llm_int8_threshold=6.0,
                                                                     llm_int8_has_fp16_weight=False)):
        super().__init__(
            model_name=model_name,
            device=device,
            lora_weights_dir=lora_weights_dir,
            lora_weights_file_override=lora_weights_file_override,
            bnb_config=bnb_config
        )
        self._feedback_examples: List[tuple[str, str]] = []

    def add_feedback_example(self, prompt: Optional[str] = None, response: Optional[str] = None) -> Optional[int]:
        """
        Adds a feedback example to the model.j
        :param prompt: The prompt to add the feedback example to.
        :param response: The response to add the feedback example to.
        :return: The number of total feedback examples.
        """
        if not prompt or not response:
            logger.warning("No prompt or response provided. Skipping feedback example.")
            return
        self._feedback_examples.append((prompt.strip(), response.strip()))
        logger.info(f"Added training example. Total examples: {len(self._feedback_examples)}")
        return len(self._feedback_examples)

    def train_on_feedback(self, output_subdir_override: Optional[str] = None) -> Optional[Path]:
        """
        Trains the model on the feedback examples and resets the feedback buffer of examples.
        :param output_subdir_override: The subdirectory to save the trained model to.
        :return: The path to the trained model.
        """
        if not self._feedback_examples:
            logger.warning("No feedback examples to train on. Skipping.")
            return None
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer must be loaded before training. It is suggested to use this within a context manager.")
        logger.info(f"Traing on {len(self._feedback_examples)} feedback examples.")
        dataset = DialoguePairDataset(self._feedback_examples, self.tokenizer)
        self.ensure_lora_adapter_loaded(error_message="LoRA adapter must be initialized or loaded before training.")
        output_path = self.fine_tune(train_dataset=dataset, output_subdir_override=output_subdir_override)
        self.reset_feedback()
        return output_path

    def reset_feedback(self):
        """
        Resets the feedback buffer.
        """
        logger.info(f"Clearing {len(self._feedback_examples)} feedback examples.")
        self._feedback_examples.clear()

    def generate_cover_letter(self, job_description: str, resume_snippets: List[str], **kwargs) -> str:
        """
        Generates a cover letter for a given job description and resume snippets.
        :param job_description: The job description to generate a cover letter for.
        :param resume_snippets: The resume snippets to use to generate the cover letter.
        :param kwargs: Additional keyword arguments to pass to the model.
        :return: The generated cover letter.
        """
        prompt = (
            "You are a helpful assistant tasked with writing a personalized cover letter.\n"
            "Use the following job description and resume snippets to create a clear, compelling narrative.\n"
            f"Job Description:\n{job_description}\n"
            f"Relevant Experience:\n{chr(10).join(resume_snippets)}\n"
            "Begin the cover letter with a warm introduction and end with a confident closing."
        )
        return self.run_prompts([prompt], **kwargs)[0]

    def assess_experience_against_posting(self, job_description: str, experience_snippets: List[str], **kwargs) -> str:
        """
        Evaluates how well the following experience matches the job description.
        :param job_description: The job description to evaluate the experience against.
        :param experience_snippets: The experience snippets to evaluate against the job description.
        :param kwargs: Additional keyword arguments to pass to the model.
        :return: The evaluation of the experience against the job description.
        """
        prompt = (
            "Evaluate how well the following experience matches the job description.\n"
            f"Job Description:\n{job_description}\n"
            f"Candidate Experience:\n{chr(10).join(experience_snippets)}\n"
            "Provide a strengths/weaknesses analysis and a 1-10 match rating."
        )
        return self.run_prompts([prompt], **kwargs)[0]

    def train_on_existing_letters(self, letter_paths: List[str], output_subdir_override: Optional[str] = None):
        """
        Trains the model on existing cover letters.
        :param letter_paths: The paths to the existing cover letters to train on.
        :param output_subdir_override: The subdirectory to save the trained model to.
        :return: The path to the trained model.
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer must be loaded before training. Use within a context manager.")
        dataset = LoraTrainingDataset(letter_paths, self.tokenizer)
        self.ensure_lora_adapter_loaded(error_message="LoRA adapter must be initialized or loaded before training.")
        return self.fine_tune(train_dataset=dataset, output_subdir_override=output_subdir_override)


class DialoguePairDataset(LoraTrainingDataset):
    def __init__(self, dialogue_pairs: List[tuple[str, str]], tokenizer: PreTrainedTokenizer,  max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs = dialogue_pairs

    def __len__(self):
        """
        Returns the number of dialogue pairs in the dataset.
        """
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        Returns the dialogue pair at the given index.
        """
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

