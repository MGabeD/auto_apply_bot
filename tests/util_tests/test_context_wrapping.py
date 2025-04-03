import types
from auto_apply_bot.utils.context_wrapping import wrap_module_methods_with_context
from tests.util_tests.mocks.mock_context_wrapping_dummy_model import DummyModel


def test_wrap_method_when_pipe_present(caplog):
    module = DummyModel(has_pipe=True)

    wrap_module_methods_with_context(module)
    result = module.context_method("foo", bar="baz")

    assert result == "executed with args=('foo',), kwargs={'bar': 'baz'}"
    assert not module.entered_context
    assert not module.exited_context
    assert "with pipe already initialized" in caplog.text


def test_wrap_method_when_pipe_missing(caplog):
    module = DummyModel(has_pipe=False)

    wrap_module_methods_with_context(module)
    result = module.context_method("hello", key="val")

    assert result == "executed with args=('hello',), kwargs={'key': 'val'}"
    assert module.entered_context
    assert module.exited_context
    assert "entering context" in caplog.text


def test_wrap_only_selected_methods():
    module = DummyModel()
    module.included = lambda: "included"
    module.excluded = lambda: "excluded"

    wrap_module_methods_with_context(module, include=["included"])
    
    assert "included" in dir(module)
    assert isinstance(getattr(module, "included"), types.MethodType)
    assert not isinstance(getattr(module, "excluded"), types.MethodType)


def test_wrap_exclude_methods():
    module = DummyModel()
    module.to_exclude = lambda: "exclude me"
    module.keep_me = lambda: "keep me"
    
    wrap_module_methods_with_context(module, exclude=["to_exclude"])
    
    assert not isinstance(getattr(module, "to_exclude"), types.MethodType)
    assert isinstance(getattr(module, "keep_me"), types.MethodType)


def test_wrap_skips_private_methods():
    module = DummyModel()
    module._private_method = lambda: "secret"
    
    wrap_module_methods_with_context(module)
    
    assert not isinstance(getattr(module, "_private_method"), types.MethodType)


def test_wrapped_method_signature_preserved():
    module = DummyModel(has_pipe=True)
    wrap_module_methods_with_context(module)
    
    result = module.context_method("a", b=2)
    assert result == "executed with args=('a',), kwargs={'b': 2}"
