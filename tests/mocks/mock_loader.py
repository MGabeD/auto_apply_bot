from langchain_core.documents import Document


class DummyLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        return [Document(page_content="This is a dummy document.", metadata={"source": self.path})]


def DummyTokenizer():
    import torch
    class _Tokenizer:
        def __call__(self, text, truncation=True, padding=False, max_length=1024, return_tensors="pt"):
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]])
            }
    return _Tokenizer()


def DummyModel():
    import torch.nn as nn
    class _Model(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.q_proj = nn.Linear(2, 2)  # target module for LoRA
            self.v_proj = nn.Linear(2, 2)
            self.model = self  # for compatibility with self.model chains
    return _Model()


class DummyPipeline:
    def __init__(self, *args, **kwargs):
        self.called_with = []
        self.model = DummyModel()  

    def __call__(self, prompts, batch_size=1, **kwargs):
        self.called_with.append((prompts, batch_size, kwargs))
        return [[{"generated_text": f"{prompt} output"}] for prompt in prompts]


def DummyPeftModel(base_model=None, peft_config=None):
    from peft import PeftModel, LoraConfig
    import torch.nn as nn
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
    class WrappedDummy(PeftModel):
        def __init__(self):
            super().__init__(base_model, peft_config)
            self.linear1 = nn.Linear(2, 2)
            self.linear2 = nn.Linear(2, 2)
        def save_pretrained(self, path): pass
    return WrappedDummy()


def dummy_from_pretrained_peft(base_model, adapter_path, **kwargs):
    return DummyPeftModel(base_model)


def patch_model_and_tokenizer(monkeypatch):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: DummyTokenizer())
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: DummyModel())
    monkeypatch.setattr("auto_apply_bot.model_interfaces.base_model_interface.pipeline", lambda *args, **kwargs: DummyPipeline())
    monkeypatch.setattr("peft.PeftModel.from_pretrained", dummy_from_pretrained_peft)

