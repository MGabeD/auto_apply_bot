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
                 mode: str = "inference",
                 bnb_config: BitsAndBytesConfig = BitsAndBytesConfig(load_in_8bit=True, 
                                                                     llm_int8_threshold=6.0,
                                                                     llm_int8_has_fp16_weight=False)):
        super().__init__(model_name, device, bnb_config)
        self.lora_weights_dir = Path(lora_weights_dir) if lora_weights_dir else resolve_project_source() / "lora_weights" / model_name.split("/")[-1]
        self.lora_weights_dir.mkdir(parents=True, exist_ok=True)
        self.lora_weights_file_override = lora_weights_file_override
        self.base_model = None
        self.model = None
        self.mode = mode

    def _load_model(self):
        self._load_base_model()
        self._load_model_with_lora_adapter()

    def __enter__(self):
        torch.cuda.empty_cache()
        log_free_memory()
        if self.mode == "inference":
            super()._load_tokenizer()
            self._load_model()
            self._load_pipeline()
        elif self.mode == "training":
            self.init_new_lora_for_training()
            self._load_pipeline()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        log_free_memory()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pipe = None
        self.model = None
        self.tokenizer = None
        self.base_model = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("Pipeline cleaned up and CUDA memory released.")

    def _load_base_model(self):
        if self.base_model is None:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                quantization_config=self.bnb_config,
                trust_remote_code=True
            )

    def _get_latest_adapter_path(self) -> Optional[Path]:
        adapter_dirs = list(self.lora_weights_dir.glob("lora_*"))
        return max(adapter_dirs, key=lambda p: p.stat().st_mtime) if adapter_dirs else None

    def _load_model_with_lora_adapter(self):
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
                self.model = self.base_model
                return
        
        self.model = PeftModel.from_pretrained(self.base_model, adapter_path)

    def _generate_lora_dirname(self) -> str:
        return datetime.now().strftime(f"lora_{self.model_name.split('/')[-1]}_%Y%m%d_%H%M%S")

    def init_new_lora_for_training(self):
        logger.info(f"Initializing new LoRA model for training: {self.model_name}")
        logger.info("Clearing CUDA memory...")
        torch.cuda.empty_cache()
        super()._load_tokenizer()
        base = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=self.bnb_config,
            trust_remote_code=True,
        )
        self.base_model = prepare_model_for_kbit_training(base)

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(base, lora_config)
        logger.info("New LoRA model initialized and assigned to self.model")

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
            data_collator=data_collator,  # 👈 Now included
            **(trainer_overrides or {})
        )

        logger.info("Starting LoRA fine-tuning...")
        trainer.train()
        self.model.save_pretrained(save_path)
        logger.info(f"LoRA adapter weights saved to {save_path}")
        self.last_trained_adapter_path = save_path
        return save_path
    
    def load_adapter(self, adapter_name: str):
        adapter_path = self.lora_weights_dir / adapter_name
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter {adapter_name} does not exist")
        if self.tokenizer is None:
            super()._load_tokenizer()
        self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
        super()._load_pipeline()

    def list_available_adapters(self) -> List[str]:
        return sorted([p.name for p in self.lora_weights_dir.glob("lora_*")])


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
