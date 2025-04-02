from auto_apply_bot.model_interfaces.base_model_interface import BaseModelInterface
from auto_apply_bot.utils.logger import get_logger
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, PreTrainedTokenizer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, prepare_model_for_kbit_training
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, List, Type, Dict, Union
from types import TracebackType
from datetime import datetime
from auto_apply_bot import resolve_project_source
from auto_apply_bot.utils.loader import load_texts_from_files
from auto_apply_bot.model_interfaces import log_free_memory


logger = get_logger(__name__)


def maybe_prepare_model(model: torch.nn.Module) -> torch.nn.Module:
    try:
        return prepare_model_for_kbit_training(model)
    except Exception as e:
        logger.warning(f"Failed to prepare model for kbit training: {e}")
        return model


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
        self.base_model: Optional[Union[AutoModelForCausalLM, PeftModel]] = None
        self.model: Optional[torch.nn.Module] = None
        self.last_loaded_adapter_path: Optional[Path] = None
        self.last_trained_adapter_path: Optional[Path] = None

    @property
    def adapter_frozen(self) -> Optional[bool]:
        """
        Returns True if all LoRA adapter parameters are frozen (not trainable),
        False if any are still trainable, or None if no LoRA adapter is loaded.
        """
        if not self.has_loaded_lora_adapter():
            return None
        return all(not param.requires_grad for param in self.model.parameters())

    def _load_model(self) -> None:
        """
        Loads the base model (quantized), stores it in base_model, and applies the most recent LoRA adapter if available.
        """
        super()._load_model()
        self.base_model = self.model
        self._load_lora_adapter()

    def __repr__(self) -> str:
        return (
            f"<LoraModelInterface(model_name={self.model_name}, "
            f"device={self.device}, "
            f"adapter_loaded={self.last_loaded_adapter_path is not None}, "
            f"adapter_frozen={self.adapter_frozen})>"
        )

    def __enter__(self) -> "LoraModelInterface":
        torch.cuda.empty_cache()
        log_free_memory()
        super()._load_tokenizer()
        self._load_model()
        self._load_pipeline()
        log_free_memory()
        return self

    def __exit__(self, exc_type: Optional[Type[Exception]], exc_val: Optional[Exception], exc_tb: Optional[TracebackType]) -> None:
        super().__exit__(exc_type, exc_val, exc_tb)
        self.base_model = None
        self.last_loaded_adapter_path = None
        self.last_trained_adapter_path = None

    def cleanup(self):
        """
        Cleans up the LoRA model interface.
        """
        logger.warning("Cleaning up LoraModelInterface's local elements before delegating to BaseModelInterface cleanup")
        self.base_model = None
        self.last_loaded_adapter_path = None
        self.last_trained_adapter_path = None
        super().cleanup()

    def _safely_attach_lora_adapter(self, adapter_path: Path) -> PeftModel:
        """
        Attempts to attach a LoRA adapter to the base model, ensuring that the adapter is properly loaded and applied.
        Raises a Runtime error w/ context if the adapter is incompatible with the base model.
        Raises a FileNotFoundError if the adapter does not exist.
        """
        if not adapter_path.exists():
            raise FileNotFoundError(f"LoRA adapter {adapter_path} does not exist.")
        try:
            model_to_return = PeftModel.from_pretrained(self.base_model, adapter_path)
            logger.info(f"Successfully attached LoRA adapter from {adapter_path}")
            return model_to_return
        except Exception as e: 
            raise RuntimeError( 
                f"Failed to attach LoRA adapter at {adapter_path} to base model '{self.model_name}'. " 
                "This may indicate architectural incompatibility or adapter corruption.\n" 
                f"Original Error: {e}"
                )

    def _get_latest_adapter_path(self) -> Optional[Path]:
        adapter_dirs = list(self.lora_weights_dir.glob("lora_*"))
        return max(adapter_dirs, key=lambda p: p.stat().st_mtime) if adapter_dirs else None

    def _load_lora_adapter(self) -> None:
        """
        Applies a LoRA adapter to the base model, using either the latest available or a user-specified adapter override. 
        This method is used during automatic model initialization and avoids reapplying an adapter if already loaded.
        """
        self._ensure_tokenizer_loaded()
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

        self.model = self._safely_attach_lora_adapter(adapter_path)
        self.last_loaded_adapter_path = adapter_path

    def _generate_lora_dirname(self) -> str:
        return datetime.now().strftime(f"lora_{self.model_name.split('/')[-1]}_%Y%m%d_%H%M%S")

    def init_new_lora_for_training(self, lora_config_override: Optional[dict] = None) -> None:
        """
        Prepares the already-loaded quantized base model for LoRA training.
        Avoids reloading the model, reducing GPU memory pressure and load time.
        """
        logger.info(f"Preparing LoRA training on loaded model: {self.model_name}")

        if self.base_model is None:
            raise RuntimeError("Base model must be loaded before initializing LoRA.")
        self.base_model = maybe_prepare_model(self.base_model)
        if "target_modules" not in (lora_config_override or {}):
            logger.warning("LoRA 'target_modules' not explicitly set. Using default ['q_proj', 'v_proj']. This may not work with all models.")
        logger.debug(f"Warning for common silent failure: target_modules={lora_config_override.get('target_modules', ['q_proj', 'v_proj'])} if you are running into issues, make sure this is correct")
        default_config = dict(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        lora_config = LoraConfig(**{**default_config, **(lora_config_override or {})})
        self.model = get_peft_model(self.base_model, lora_config)
        logger.info("LoRA layers initialized and attached to the base model.")
        self._load_pipeline()

    def fine_tune(self, 
              train_dataset: Dataset, 
              output_subdir_override: Optional[str] = None, 
              training_args_override: Optional[dict] = None, 
              trainer_overrides: Optional[dict] = None) -> Path:
        """
        Fine-tunes the current LoRA model on a dataset using PEFT and HuggingFace Trainer.
        Saves to a subdirectory under the lora_weights_dir.
        :param train_dataset: The dataset to train on.
        :param output_subdir_override: The optional subdirectory override for where to save the fine-tuned model to.
        :param training_args_override: Additional training arguments to pass to the Trainer.
        :param trainer_overrides: Additional trainer arguments to pass to the Trainer.
        :return: The path to the saved fine-tuned model.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before fine-tuning. Use within a context manager.")

        self.ensure_lora_adapter_loaded()

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
    
    def load_adapter(self, adapter_name: str, force_reload: bool = False) -> None:
        """
        Loads a specific LoRA adapter by name and wraps it around the base model. Skips reloading if the requested adapter is already active. 
        This method is designed for runtime adapter swapping after the initial model load.
        """
        adapter_path = self.lora_weights_dir / adapter_name
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter {adapter_name} does not exist")
        if not force_reload and self.last_loaded_adapter_path == adapter_path:
            logger.info(f"Adapter {adapter_name} is already loaded. Skipping reload.")
            return
        self._ensure_tokenizer_loaded()
        self.model = self._safely_attach_lora_adapter(adapter_path)
        self.last_loaded_adapter_path = adapter_path
        super()._load_pipeline()

    def list_available_adapters(self) -> List[str]:
        return sorted([p.name for p in self.lora_weights_dir.glob("lora_*")])

    def continue_training(self, train_dataset: Dataset, save_to: Optional[str] = None, **kwargs) -> Path:
        """
        Continues training on the currently loaded LoRA adapter.
        Saves to the given subdirectory or reuses the last one used.
        """
        self.ensure_lora_adapter_loaded()

        save_name = save_to or (self.last_trained_adapter_path.name if self.last_trained_adapter_path else None)
        if not save_name:
            raise ValueError("No target directory specified and no previous training directory found.")
        logger.info(f"Continuing training on {self.last_loaded_adapter_path} with dataset: {train_dataset}")
        return self.fine_tune(train_dataset, output_subdir_override=save_name, **kwargs)
    
    def has_loaded_lora_adapter(self) -> bool:
        """
        Checks if a LoRA adapter has been loaded.
        Returns True if a LoRA adapter is loaded and False otherwise.
        """
        return isinstance(self.model, PeftModel) and self.last_loaded_adapter_path is not None
    
    def ensure_lora_adapter_loaded(self, error_message: str = "No LoRA adapter is currently loaded.") -> None:
        if not self.has_loaded_lora_adapter():
            raise RuntimeError(error_message)

    def _ensure_tokenizer_loaded(self) -> None:
        if self.tokenizer is None:
            super()._load_tokenizer()

    def freeze_lora_adapter(self) -> None:
        """
        Freezes the LoRA adapter weights, preventing them from being updated during training.
        """
        self.ensure_lora_adapter_loaded()
        if isinstance(self.model, PeftModel):
            for param in self.model.parameters():
                param.requires_grad = False
            logger.info("LoRA adapter weights frozen.")
        else:
            logger.warning("Model is not a PeftModel, skipping freezing.")

    def unfreeze_lora_adapter(self) -> None:
        """
        Unfreezes the LoRA adapter weights, allowing them to be updated during training.
        """
        self.ensure_lora_adapter_loaded()
        if isinstance(self.model, PeftModel):
            for param in self.model.parameters():
                param.requires_grad = True
            logger.info("LoRA adapter weights unfrozen.")
        else:
            logger.warning("Model is not a PeftModel, skipping unfreeze.")

    def get_adapter_config(self) ->  Dict[str, Optional[str] | Optional[bool] | List[str]]:
        """
        Returns the metadata for the currently loaded LoRA adapter.
        """
        return {
            "loaded_adapter": str(self.last_loaded_adapter_path) if self.last_loaded_adapter_path else None,
            "last_trained_adapter": str(self.last_trained_adapter_path) if self.last_trained_adapter_path else None,
            "adapter_frozen": self.adapter_frozen,
            "available_adapters": self.list_available_adapters()
        }
    
    def reset_lora_adapter(self) -> None:
        """
        Resets the LoRA adapter to the base model.
        """
        if self.base_model is None:
            raise RuntimeError("Base model must be loaded before resetting LoRA adapter.")
        if self.model == self.base_model:
            logger.info("Model is already the base model. Skipping reset.")
            return
        self.model = self.base_model
        self.last_loaded_adapter_path = None
        self.last_trained_adapter_path = None
        logger.info("LoRA adapter reset to base model.")
        self._load_pipeline()


class LoraTrainingDataset(Dataset):
    def __init__(self, file_paths: List[str], tokenizer: PreTrainedTokenizer, max_length: Optional[int] = None) -> None:
        self.samples = []
        max_length = max_length or tokenizer.model_max_length
        raw_texts = load_texts_from_files(file_paths)
        for text in raw_texts:
            encoded = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
            self.samples.append({
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "labels": encoded["input_ids"].squeeze(0).clone()
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]
