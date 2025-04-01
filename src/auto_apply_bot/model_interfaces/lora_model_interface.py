from auto_apply_bot.model_interfaces.base_model_interface import BaseModelInterface
from auto_apply_bot.logger import get_logger
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, pipeline, PreTrainedTokenizer, DataCollatorForLanguageModeling    
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, prepare_model_for_kbit_training
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from auto_apply_bot import resolve_project_source
from auto_apply_bot.loader import load_texts_from_files
from auto_apply_bot.model_interfaces import determine_batch_size, log_free_memory


logger = get_logger(__name__)


class LoraModelInterface(BaseModelInterface):
    def __init__(self,
                 model_name: str = "deepseek-ai/deepseek-llm-7b-chat",
                 device: str = "cuda",
                 lora_weights_dir: Optional[str] = None,
                 lora_weights_file_override: Optional[str] = None,
                 bnb_config: BitsAndBytesConfig = BitsAndBytesConfig(load_in_8bit=True, 
                                                                     llm_int8_threshold=6.0,
                                                                     llm_int8_has_fp16_weight=False)):
        super().__init__(model_name, device, bnb_config)
        self.lora_weights_dir = Path(lora_weights_dir) if lora_weights_dir else resolve_project_source() / "lora_weights" / model_name.split("/")[-1]
        self.lora_weights_dir.mkdir(parents=True, exist_ok=True)
        self.lora_weights_file_override = lora_weights_file_override
        self.base_model = None
        self.model = None
        self.last_loaded_adapter_path = None
        self.last_trained_adapter_path = None

    @property
    def is_adapter_frozen(self) -> Optional[bool]:
        """
        Returns True if all LoRA adapter parameters are frozen (not trainable),
        False if any are still trainable, or None if no LoRA adapter is loaded.
        """
        if not self.has_loaded_lora_adapter():
            return None
        return all(not param.requires_grad for param in self.model.parameters())

    def _load_model(self):
        """
        Loads the base model (quantized), stores it in base_model, and applies the most recent LoRA adapter if available.
        """
        super()._load_model()
        self.base_model = self.model
        self._load_lora_adapter()

    def __enter__(self):
        torch.cuda.empty_cache()
        log_free_memory()
        super()._load_tokenizer()
        self._load_model()
        self._load_pipeline()
        log_free_memory()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pipe = None
        self.model = None
        self.tokenizer = None
        self.base_model = None
        self.last_loaded_adapter_path = None
        self.last_trained_adapter_path = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("Pipeline cleaned up and CUDA memory released.")

    def _get_latest_adapter_path(self) -> Optional[Path]:
        adapter_dirs = list(self.lora_weights_dir.glob("lora_*"))
        return max(adapter_dirs, key=lambda p: p.stat().st_mtime) if adapter_dirs else None

    def _load_lora_adapter(self):
        """
        Applies a LoRA adapter to the base model, using either the latest available or a user-specified adapter override. 
        This method is used during automatic model initialization and avoids reapplying an adapter if already loaded.
        """
        if self.lora_weights_file_override:
            adapter_path = self.lora_weights_dir / self.lora_weights_file_override
            if not adapter_path.exists():
                raise FileNotFoundError(f"LoRA override path {adapter_path} does not exist.")
            logger.info(f"Using override LoRA adapter from {adapter_path}")
        else:
            adapter_path = self._get_latest_adapter_path()
            if adapter_path:
                logger.info(f"Loading latest LoRA adapter from {adapter_path}")
            else:
                logger.warning("No LoRA adapter found. Using base model only.")
                return

        if self.last_loaded_adapter_path == adapter_path:
            logger.info(f"Adapter {adapter_path} is already loaded. Skipping reload.")
            return

        self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
        self.last_loaded_adapter_path = adapter_path

    def _generate_lora_dirname(self) -> str:
        return datetime.now().strftime(f"lora_{self.model_name.split('/')[-1]}_%Y%m%d_%H%M%S")

    def init_new_lora_for_training(self):
        """
        Prepares the already-loaded quantized base model for LoRA training.
        Avoids reloading the model, reducing GPU memory pressure and load time.
        """
        logger.info(f"Preparing LoRA training on loaded model: {self.model_name}")

        if self.base_model is None:
            raise RuntimeError("Base model must be loaded before initializing LoRA.")

        self.base_model = prepare_model_for_kbit_training(self.base_model)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.base_model, lora_config)
        logger.info("LoRA layers initialized and attached to the base model.")
        self._load_pipeline()

    def fine_tune(self, 
              train_dataset, 
              output_subdir_override: Optional[str] = None, 
              training_args_override: Optional[dict] = None, 
              trainer_overrides: Optional[dict] = None) -> Path:
        """
        Fine-tunes the current LoRA model on a dataset using PEFT and HuggingFace Trainer.
        Saves to a subdirectory under the lora_weights_dir.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before fine-tuning. Use within a context manager.")

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

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  
        )

        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,  
            **(trainer_overrides or {})
        )

        logger.info("Starting LoRA fine-tuning...")
        trainer.train()
        self.model.save_pretrained(save_path)
        logger.info(f"LoRA adapter weights saved to {save_path}")
        self.last_trained_adapter_path = save_path
        return save_path
    
    def load_adapter(self, adapter_name: str):
        """
        Loads a specific LoRA adapter by name and wraps it around the base model. Skips reloading if the requested adapter is already active. 
        This method is designed for runtime adapter swapping after the initial model load.
        """
        adapter_path = self.lora_weights_dir / adapter_name
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter {adapter_name} does not exist")
        if self.last_loaded_adapter_path == adapter_path:
            logger.info(f"Adapter {adapter_name} is already loaded. Skipping reload.")
            return
        if self.tokenizer is None:
            super()._load_tokenizer()
        self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
        self.last_loaded_adapter_path = adapter_path
        super()._load_pipeline()

    def list_available_adapters(self) -> List[str]:
        return sorted([p.name for p in self.lora_weights_dir.glob("lora_*")])

    def continue_training(self, train_dataset, save_to: Optional[str] = None, **kwargs) -> Path:
        """
        Continues training on the currently loaded LoRA adapter.
        Saves to the given subdirectory or reuses the last one used.
        """
        if not isinstance(self.model, PeftModel):
            raise RuntimeError("No LoRA adapter is currently loaded.")

        save_name = save_to or (self.last_trained_adapter_path.name if self.last_trained_adapter_path else None)
        if not save_name:
            raise ValueError("No target directory specified and no previous training directory found.")

        return self.fine_tune(train_dataset, output_subdir_override=save_name, **kwargs)
    
    def has_loaded_lora_adapter(self) -> bool:
        """
        Checks if a LoRA adapter has been loaded.
        Returns True if a LoRA adapter is loaded and False otherwise.
        """
        return isinstance(self.model, PeftModel) and self.last_loaded_adapter_path is not None
    
    def freeze_lora_adapter(self):
        """
        Freezes the LoRA adapter weights, preventing them from being updated during training.
        """
        if not self.has_loaded_lora_adapter():
            raise RuntimeError("No LoRA adapter is currently loaded.")
        if isinstance(self.model, PeftModel):
            for param in self.model.parameters():
                param.requires_grad = False
            logger.info("LoRA adapter weights frozen.")
        else:
            logger.warning("Model is not a PeftModel, skipping freezing.")

    def unfreeze_lora_adapter(self):
        """
        Unfreezes the LoRA adapter weights, allowing them to be updated during training.
        """
        if not self.has_loaded_lora_adapter():
            raise RuntimeError("No LoRA adapter is currently loaded.")
        if isinstance(self.model, PeftModel):
            for param in self.model.parameters():
                param.requires_grad = True
            logger.info("LoRA adapter weights unfrozen.")
        else:
            logger.warning("Model is not a PeftModel, skipping unfrozen.")


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
