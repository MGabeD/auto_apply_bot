from typing import List, Any, Callable, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from auto_apply_bot.logger import get_logger
import torch
from accelerate import infer_auto_device_map, init_empty_weights
from auto_apply_bot.model_interfaces import determine_batch_size, log_free_memory


logger = get_logger(__name__)


class BaseModelInterface:
    def __init__(self,
                 model_name: str = "deepseek-ai/deepseek-llm-7b-chat",
                 device: str = "cuda",
                 bnb_config: BitsAndBytesConfig = BitsAndBytesConfig(load_in_8bit=True,
                                                                     llm_int8_threshold=6.0,
                                                                     llm_int8_has_fp16_weight=False)):
        """
        Base model interface to handle LLM loading and prompt execution.
        :param model_name: Hugging Face model name or path
        :param device: Target device ('cuda' or 'cpu')
        :param bnb_config: Optional quantization config (e.g., BitsAndBytesConfig)
        """
        self.model_name = model_name
        self.active_model = model_name
        self.device = device
        self.bnb_config = bnb_config
        self.pipe = None
        self.tokenizer = None

    def _load_tokenizer(self):
         if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.active_model, use_fast=True)

    def _load_model(self) -> AutoModelForCausalLM:
        try:
            self.model = self._load_gpu_only()
        except Exception as e:
            logger.warning(f"Falling back to GPU+CPU split due to: {e}")
            self.model = self._load_with_fallback()

    def _load_pipeline(self):
        """
        This _load_pipeline function is used to load the pipeline for the model. It can work with both regular models which are supported by Huggingface
        and extract the .model from the model if it is a peft model. Thus, we can run both HuggingFace compatible models and peft models with a single funciton for child classes.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before initializing pipeline.")
        self.pipe = pipeline("text-generation", model=getattr(self.model, "model", self.model), tokenizer=self.tokenizer)

    def __enter__(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()
        log_free_memory()
        self._load_tokenizer()
        self._load_model()
        self._load_pipeline()
        log_free_memory()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pipe = None
        self.tokenizer = None
        self.model = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("Pipeline cleaned up and CUDA memory released.")

    def _load_gpu_only(self):
        logger.info("Attempting full GPU load...")
        model = AutoModelForCausalLM.from_pretrained(
            self.active_model,
            device_map={"": 0} if self.device == "cuda" else {"": "cpu"},
            quantization_config=self.bnb_config,
            trust_remote_code=True
        )
        logger.info("Loaded fully on GPU.")
        return model

    def _load_with_fallback(self):
        with init_empty_weights():
            model_init = AutoModelForCausalLM.from_pretrained(
                self.active_model,
                quantization_config=self.bnb_config,
                trust_remote_code=True
            )
        device_map = infer_auto_device_map(
            model_init,
            max_memory={0: "16GiB", "cpu": "64GiB"},
            no_split_module_classes=["DecoderLayer"]
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.active_model,
            device_map=device_map,
            quantization_config=self.bnb_config,
            trust_remote_code=True
        )
        logger.warning(f"Loaded model with CPU fallback. Device map: {device_map}")
        return model

    def run_prompts(self, prompts: List[str], batch_size_override: Optional[int] = None,
                    post_process_fn: Optional[Callable[[str], Any]] = None, **generation_kwargs) -> List[Any]:
        """
        Executes a list of prompts using the loaded pipeline.
        :param prompts: List of prompts
        :param batch_size_override: Optional batch size override
        :param post_process_fn: Optional function to post-process generated text
        :param generation_kwargs: Hugging Face generation parameters
        :return: List of generated texts
        """
        logger.info(f"Pipeline initialized with: {type(self.pipe.model).__name__}")
        batch_size = batch_size_override if batch_size_override is not None else determine_batch_size()
        if not self.pipe:
            raise RuntimeError("Pipeline is not initialized. Use within a context manager.")

        logger.info(f"Running {len(prompts)} prompts with batch size {batch_size}.")
        outputs = self.pipe(
            prompts,
            batch_size=batch_size,
            **generation_kwargs
        )
        logger.info(f"Generated {len(outputs)} outputs.")
        if post_process_fn:
            return [post_process_fn(o[0]["generated_text"]) for o in outputs]
        else:
            # logger.info(f"Post-processing outputs... FIND ERRORS HERE MODEL INTERFACE IS CORECT")
            # for i,j in zip(outputs, prompts):
            #     logger.warning(f"Output: {i[0]['generated_text']}")
            #     logger.warning(f"filtered output: {i[0]['generated_text'].replace(j, '').strip()}")
            return [
                o[0]["generated_text"].replace(prompt, "").strip()
                for o, prompt in zip(outputs, prompts)
            ]

