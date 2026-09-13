from .base import Backend, StepResult
from .fake import FakeBackend

__all__ = ["Backend", "StepResult", "FakeBackend", "ReferenceBackend", "expert_placement"]


def __getattr__(name):
    if name == "ReferenceBackend":
        from .reference import ReferenceBackend

        return ReferenceBackend
    if name == "expert_placement":
        # importlib, not `from . import ...`: the latter resolves the name through
        # this module's __getattr__ and recurses. Cache it so later lookups are free.
        import importlib

        mod = importlib.import_module(".expert_placement", __package__)
        globals()[name] = mod
        return mod
    raise AttributeError(name)
