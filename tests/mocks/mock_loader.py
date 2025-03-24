from langchain_core.documents import Document

class DummyLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        return [Document(page_content="This is a dummy document.", metadata={"source": self.path})]
