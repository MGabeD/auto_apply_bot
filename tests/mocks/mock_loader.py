from langchain_core.documents import Document
from peft import PeftModel, LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch.nn as nn
import torch


class DummyLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        return [Document(page_content="This is a dummy document.", metadata={"source": self.path})]


class DummyTokenizer:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, text, truncation=True, padding=False, max_length=1024, return_tensors="pt"):
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }


class DummyModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.q_proj = nn.Linear(2, 2)  # target module for LoRA
        self.v_proj = nn.Linear(2, 2)
        self.model = self  # for compatibility with self.model.model chains


class DummyPipeline:
    def __init__(self, *args, **kwargs):
        self.called_with = []
        self.model = DummyModel()  

    def __call__(self, prompts, batch_size=1, **kwargs):
        self.called_with.append((prompts, batch_size, kwargs))
        return [[{"generated_text": f"{prompt} output"}] for prompt in prompts]


class DummyPeftModel(PeftModel):
    def __init__(self, base_model=None, peft_config=None):
        if base_model is None:
            base_model = DummyModel()
        if peft_config is None:
            peft_config = LoraConfig(
                r=4,
                lora_alpha=8,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
            )
        super().__init__(base_model, peft_config)
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 2)

    def save_pretrained(self, path):
        pass


def dummy_from_pretrained_peft(base_model, adapter_path, **kwargs):
    return DummyPeftModel(base_model)


def patch_model_and_tokenizer(monkeypatch):
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: DummyTokenizer())
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: DummyModel())
    monkeypatch.setattr("auto_apply_bot.model_interfaces.base_model_interface.pipeline", lambda *args, **kwargs: DummyPipeline())
    monkeypatch.setattr("peft.PeftModel.from_pretrained", dummy_from_pretrained_peft)

