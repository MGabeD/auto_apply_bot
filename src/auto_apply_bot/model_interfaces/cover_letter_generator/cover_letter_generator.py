from types import TracebackType
from auto_apply_bot.model_interfaces.lora_model_interface import LoraModelInterface, LoraTrainingDataset
from auto_apply_bot.loader import load_texts_from_files
from auto_apply_bot.logger import get_logger
from transformers import PreTrainedTokenizer, BitsAndBytesConfig
from typing import List, Optional, Type, Callable
from pathlib import Path


logger = get_logger(__name__)


def default_formatter(prompt: str, response: str):
    return f"### Prompt:\n{prompt.strip()}\n\n### Response:\n{response.strip()}"


class CoverLetterModelInterface(LoraModelInterface):
    
    def __init__(self,
                 model_name: str = "deepseek-ai/deepseek-llm-7b-chat",
                 device: str = "cuda",
                 lora_weights_dir: Optional[str] = None,
                 lora_weights_file_override: Optional[str] = None,
                 formatter: Callable[[str, str], str] = default_formatter,
                 bnb_config: BitsAndBytesConfig = BitsAndBytesConfig(load_in_8bit=True,
                                                                     llm_int8_threshold=6.0,
                                                                     llm_int8_has_fp16_weight=False)):
        super().__init__(
            model_name=model_name,
            device=device,
            lora_weights_dir=lora_weights_dir,
            lora_weights_file_override=lora_weights_file_override,
            bnb_config=bnb_config,
        )
        self._feedback_examples: List[tuple[str, str]] = []
        self.formatter = formatter

    # MARK: This section is for handling LoRA training for the cover letter generator

    # MARK: This sub-section is for handling feedback examples for the cover letter generator
    def add_feedback_example(self, prompt: Optional[str] = None, response: Optional[str] = None) -> Optional[int]:
        """
        Adds a feedback example to the model.
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

    def train_on_dialogue_pairs(self, 
                                dialogue_pairs: Optional[List[tuple[str, str]]] = None,
                                load_from_buffer: bool = False,
                                output_subdir_override: Optional[str] = None) -> Optional[Path]:
        """
        Trains the model on the dialogue pairs and resets the dialogue pairs buffer of examples.
        :param dialogue_pairs: The dialogue pairs to train on.
        :param load_from_buffer: Whether to load the dialogue pairs from the buffer.
        :param output_subdir_override: The subdirectory to save the trained model to.
        :return: The path to the trained model.
        """
        if not load_from_buffer and dialogue_pairs is None:
            raise ValueError("Either dialogue_pairs must be provided or load_from_buffer must be True.")
        train_data = dialogue_pairs or []
        if load_from_buffer:
            train_data = train_data + self._feedback_examples
        if len(train_data) < 1:
            logger.warning("No feedback examples to train on. Skipping.")
            return None
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer must be loaded before training. It is suggested to use this within a context manager.")
        logger.info(f"Training on {len(train_data)} dialogue pairs.")
        dataset = DialoguePairDataset(train_data, self.tokenizer)
        self.ensure_lora_adapter_loaded(error_message="LoRA adapter must be initialized or loaded before training.")
        output_path = self.fine_tune(train_dataset=dataset, output_subdir_override=output_subdir_override)
        if load_from_buffer:
            self.reset_feedback()
        return output_path

    def reset_feedback(self):
        """
        Resets the feedback buffer.
        """
        logger.info(f"Clearing {len(self._feedback_examples)} feedback examples.")
        self._feedback_examples.clear()

    # MARK: This sub-section is for handling training on existing writing samples
    def train_on_existing_letters(self, letter_paths: List[str], output_subdir_override: Optional[str] = None) -> Optional[Path]:
        """
        Trains the model on existing cover letters by wrapping them with a generic prompt.
        :param letter_paths: The paths to the existing cover letters to train on.
        :param output_subdir_override: The subdirectory to save the trained model to.
        :return: The path to the trained model.
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer must be loaded before training. Use within a context manager.")
        raw_letters = load_texts_from_files(letter_paths)
        dialogue_pairs = [
            ("Write a high-quality, personalized cover letter based on my experience.", letter.strip())
            for letter in raw_letters if letter.strip()
        ]

        if not dialogue_pairs:
            logger.warning("No valid cover letters found. Skipping training.")
            return None

        logger.info(f"Loaded {len(dialogue_pairs)} cover letters as training pairs.")
        return self.train_on_dialogue_pairs(dialogue_pairs=dialogue_pairs, output_subdir_override=output_subdir_override)

    # MARK: This __exit__ adds in a training session for RHLF training if there are any feedback examples
    def __exit__(self, exc_type: type[Exception] | None, exc_val: Exception | None, exc_tb: TracebackType | None) -> None:
        """
        Exits the context manager and trains on the dialogue pairs.
        """
        if self._feedback_examples:
            # Defensive check - yes it adds more depth but will be less error prone
            try:
                self.train_on_dialogue_pairs(load_from_buffer=True)
            except Exception as e:
                logger.error(f"Error training on dialogue pairs: {e}, Didn't train on dialogue pairs.")
        return super().__exit__(exc_type, exc_val, exc_tb)

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
        wrapped = self.formatter(prompt, "")
        return self.run_prompts([wrapped], **kwargs)[0]

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


class DialoguePairDataset(LoraTrainingDataset):
    def __init__(self, dialogue_pairs: List[tuple[str, str]], tokenizer: PreTrainedTokenizer,  max_length: int = 1024, formatter: Callable[[str, str], str] = default_formatter):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs = dialogue_pairs
        self.formatter = formatter

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
        text = self.formatter(prompt, response)
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

