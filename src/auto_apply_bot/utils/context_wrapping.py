import types
import functools
from auto_apply_bot.utils.logger import get_logger


logger = get_logger(__name__)


def wrap_module_methods_with_context(module, context_attr: str = "pipe", include: list[str] = None, exclude: list[str] = None):
    """
    Wraps all public methods of a module to automatically enter/exit its context if the context attribute is not set.
    :param module: The module (object) whose methods to wrap.
    :param context_attr: The attribute that determines if the module is already initialized (e.g., "pipe").
    :param include: Optional list of method names to include. If None, include all public methods.
    :param exclude: Optional list of method names to exclude.
    """
    method_names = [
        attr for attr in dir(module)
        if callable(getattr(module, attr))
        and not attr.startswith("_")
        and (include is None or attr in include)
        and (exclude is None or attr not in exclude)
    ]

    for name in method_names:
        original = getattr(module, name)

        @functools.wraps(original)
        def wrapper(*args, _original=original, _module=module, _name=name, **kwargs):
            context_ready = getattr(_module, context_attr, None) is not None
            if context_ready:
                logger.info(f"Calling {_module.__class__.__name__}.{_name} with pipe already initialized.")
                return _original(*args, **kwargs)
            logger.info(f"Calling {_module.__class__.__name__}.{_name} with pipe not initialized, entering context.")
            with _module:
                return _original(*args, **kwargs)

        setattr(module, name, types.MethodType(wrapper, module))
        logger.debug(f"Wrapped `{module.__class__.__name__}.{name}` with context manager fallback.")
