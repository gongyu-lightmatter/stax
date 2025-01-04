from typing import Protocol, runtime_checkable

from .saxtypes import SType


@runtime_checkable
class StatefulModelBuilder(Protocol):
    def __init__(self): ...

    def __call__(self, *args, **kwargs) -> SType: ...
