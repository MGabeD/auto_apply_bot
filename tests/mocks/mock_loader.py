from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class DummyLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        return [Document(page_content="This is a dummy document.", metadata={"source": self.path})]


class DummyTokenizer:
    def __init__(self, *args, **kwargs):
        pass


class DummyModel:
    def __init__(self, *args, **kwargs):
        self.model = self  


class DummyPipeline:
    def __init__(self, *args, **kwargs):
        self.called_with = []
        self.model = DummyModel()  

    def __call__(self, prompts, batch_size=1, **kwargs):
        self.called_with.append((prompts, batch_size, kwargs))
        return [[{"generated_text": f"{prompt} output"}] for prompt in prompts]


def patch_model_and_tokenizer(monkeypatch):
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: DummyTokenizer())
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: DummyModel())
    monkeypatch.setattr("auto_apply_bot.model_interfaces.base_model_interface.pipeline", lambda *args, **kwargs: DummyPipeline())
