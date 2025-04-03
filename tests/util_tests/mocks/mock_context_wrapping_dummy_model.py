class DummyModel:
    def __init__(self, has_pipe=True):
        self.pipe = "fake_pipe" if has_pipe else None
        self.entered_context = False
        self.exited_context = False

    def context_method(self, *args, **kwargs):
        return f"executed with args={args}, kwargs={kwargs}"

    def __enter__(self):
        self.entered_context = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exited_context = True

    def context_method(self, *args, **kwargs):
        args_no_self = args[1:] if args and isinstance(args[0], DummyModel) else args
        return f"executed with args={args_no_self}, kwargs={kwargs}"