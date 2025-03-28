from auto_apply_bot.model_interfaces.base_model_interface import BaseModelInterface
from auto_apply_bot.logger import get_logger
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, pipeline, PreTrainedTokenizer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from peft.tuners.lora import prepare_model_for_kbit_training
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from auto_apply_bot import resolve_project_source
from auto_apply_bot.loader import load_texts_from_files


logger = get_logger(__name__)


class LoraModelInterface(BaseModelInterface):
    def __init__(self,
                 model_name: str = "deepseek-ai/deepseek-llm-7b-chat",
                 device: str = "cuda",
                 lora_weights_dir: Optional[str] = None,
                 bnb_config: BitsAndBytesConfig = BitsAndBytesConfig(load_in_8bit=True)):
        super().__init__(model_name, device, bnb_config)
        self.lora_weights_dir = Path(lora_weights_dir) if lora_weights_dir else resolve_project_source() / "lora_weights" / model_name.split("/")[-1]
        self.lora_weights_dir.mkdir(parents=True, exist_ok=True)

    def _generate_lora_dirname(self) -> str:
        return datetime.now().strftime(f"lora_{self.model_name.split('/')[-1]}_%Y%m%d_%H%M%S")

    def prepare_lora_model(self):
        logger.info(f"Preparing LoRA model for {self.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            device_map="auto",
            quantization_config=self.bnb_config,
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

        logger.info(f"LoRA model prepared for {self.model_name}")
        return model

    def fine_tune(self,
                  model, 
                  tokenizer,
                  train_dataset,
                  output_subdir_override: Optional[str] = None,
                  training_args_override: Optional[dict] = None,
                  trainer_overrides: Optional[dict] = None,
                  ) -> Path:
        logger.info(f"Fine-tuning {self.model_name}")
        output_name = output_subdir_override or self._generate_lora_dirname()
        save_path = self.lora_weights_dir / output_name
        save_path.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(save_path),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=1,
            learning_rate=5e-5,
            fp16=torch.cuda.is_available(),
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

        logger.info("Starting LoRA fine-tuning...")
        trainer.train()
        model.save_pretrained(save_path)
        logger.info(f"LoRA adapter weights saved to {save_path}")
        return save_path

    def load_or_prepare_inference_pipeline(self, lora_weights_path: Optional[str] = None):
        """
        Loads the model pipeline. If LoRA weights are provided, loads the adapter into the base model.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=self.bnb_config,
            trust_remote_code=True
        )

        if lora_weights_path and Path(lora_weights_path).exists():
            logger.info(f"Loading LoRA adapter from {lora_weights_path}")
            model = PeftModel.from_pretrained(base_model, lora_weights_path)
        else:
            logger.warning("No LoRA adapter provided or found. Running base model only.")
            model = base_model

        self.pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if self.device == "cuda" else -1)
        logger.info("Model pipeline initialized.")


class LoraTrainingDataset(Dataset):
    def __init__(self, file_paths: List[str], tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.samples = load_texts_from_files(file_paths)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }