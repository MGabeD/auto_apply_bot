from auto_apply_bot.model_interfaces.base_model_interface import BaseModelInterface
from auto_apply_bot.utils.logger import get_logger
from auto_apply_bot import resolve_project_source
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, pipeline
import torch
from torch.utils.data import Dataset
from typing import Optional
from pathlib import Path
from datetime import datetime


logger = get_logger(__name__)


class FineTunableModelInterface(BaseModelInterface):

    def __init__(self,
                 model_name: str = "deepseek-ai/deepseek-llm-7b-chat",
                 device: str = "cuda",
                 local_fine_tuned_model_dir_override: Optional[str] = None,
                 local_fine_tune_model_path_override: Optional[str] = None,
                 bnb_config: BitsAndBytesConfig = BitsAndBytesConfig(load_in_8bit=True, 
                                                                     llm_int8_threshold=6.0,
                                                                     llm_int8_has_fp16_weight=False)):
        super().__init__(model_name, device, bnb_config)
        if local_fine_tuned_model_dir_override is not None:
            self.local_fine_tuned_model_dir = Path(local_fine_tuned_model_dir_override)
        else:
            project_root = resolve_project_source()
            self.local_fine_tuned_model_dir = project_root / "models" / model_name.split("/")[-1]
        self.local_fine_tuned_model_dir.mkdir(parents=True, exist_ok=True)
        self.local_fine_tune_model_path_override = local_fine_tune_model_path_override
        self.active_model = self._get_latest_model_path() or model_name

    def _generate_model_filename(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"tuned_{timestamp}"    

    def _get_latest_model_path(self) -> Optional[Path]:
        if self.local_fine_tune_model_path_override is not None:
            return Path(self.local_fine_tune_model_path_override)
        candidate_files = list(self.local_fine_tuned_model_dir.glob("*"))
        model_dirs = [c for c in candidate_files if c.is_dir()]
        if not model_dirs:
            logger.warning(f"No fine-tuned models found in {self.local_fine_tuned_model_dir}")
            return None
        latest = max(model_dirs, key=lambda x: x.stat().st_mtime)
        logger.info(f"Latest fine-tuned model: {latest}")
        return latest 
        
    def save_model(self,
               model: torch.nn.Module,
               tokenizer: AutoTokenizer,
               model_save_dir_override: Optional[str] = None,
               fine_tuned_model_name_override: Optional[str] = None) -> Path:
        """
        Saves the model and tokenizer to the appropriate directory.
        If no override is provided, saves into self.local_fine_tuned_model_dir / generated timestamp folder.
        :param model: The model to save
        :param tokenizer: The tokenizer to save
        :param model_save_dir_override: Optional directory to save the model
        :param fine_tuned_model_name_override: Optional name for the folder inside the directory
        :return: Path to the saved model directory
        """
        save_dir = Path(model_save_dir_override) if model_save_dir_override else self.local_fine_tuned_model_dir
        save_subdir_name = fine_tuned_model_name_override or self._generate_model_filename()
        
        save_path = save_dir / save_subdir_name
        save_path.mkdir(parents=True, exist_ok=True)

        try:
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            logger.info(f"Model and tokenizer saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving model to {save_path}: {e}", exc_info=True)
            raise
        return save_path

    def fine_tune(self,
                  model: AutoModelForCausalLM,
                  tokenizer: AutoTokenizer,
                  train_dataset: Dataset,
                  model_save_dir_override: Optional[str] = None,
                  fine_tuned_model_name_override: Optional[str] = None,
                  training_args_override: Optional[dict] = None,
                  trainer_overrides: Optional[dict] = None) -> Path:
        """
        Fine-tunes the current model on the provided dataset and saves the resulting model and tokenizer.
        :param model: The model to fine-tune must be "AutoModelForCausalLM"
        :param tokenizer: The tokenizer to fine-tune must be "AutoTokenizer"
        :param train_dataset: The training dataset must be "Dataset"
        :param output_subdir: Optional subdirectory to save the fine-tuned model
        :param training_args_override: Optional overrides for the training arguments
        :param trainer_overrides: Optional overrides for the trainer
        :return: Path to the saved model directory
        """
        training_args = TrainingArguments(
            output_dir=str(save_path), 
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=1,
            learning_rate=5e-5,
            fp16=True if torch.cuda.is_available() else False,
            logging_dir=str(save_path / "logs"),
            save_strategy="no",
            report_to="none",
            logging_steps=10,
            **(training_args_override or {})
        )
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            **(trainer_overrides or {})
        )

        logger.info(f"Starting training for model {self.model_name}...")
        trainer.train()
        logger.info("Training complete. Saving...")

        save_path = self.save_model(model, tokenizer, fine_tuned_model_name_override=fine_tuned_model_name_override, model_save_dir_override=model_save_dir_override)
        logger.info(f"Model and tokenizer saved to {save_path}")
        return save_path

    def reload_pipe_from_latest(self):
        latest_model_path = self._get_latest_model_path()
        if latest_model_path is None:
            raise FileNotFoundError("No fine-tuned model found to reload.")
        tokenizer = AutoTokenizer.from_pretrained(latest_model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(latest_model_path, trust_remote_code=True)
        self.pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if self.device == "cuda" else -1)

    def cleanup(self):
        """
        Cleans up the fine-tuning base model interface.
        """
        logger.warning("Cleaning up FineTunableModelInterface - identical to BaseModelInterface cleanup")
        super().cleanup()

