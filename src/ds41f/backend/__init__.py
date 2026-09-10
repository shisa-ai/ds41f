from .base import Backend, StepResult
from .fake import FakeBackend

__all__ = ["Backend", "StepResult", "FakeBackend", "ReferenceBackend"]

def __getattr__(name):
    if name == "ReferenceBackend":
        from .reference import ReferenceBackend

        return ReferenceBackend
    raise AttributeError(name)
