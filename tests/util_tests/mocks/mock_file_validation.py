from io import BytesIO


class DummyFile(BytesIO):
    def __init__(self, content, name="file.txt", size=None):
        super().__init__(content)
        self.name = name
        self.size = size if size is not None else len(content)

        
