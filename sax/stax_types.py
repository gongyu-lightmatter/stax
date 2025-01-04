from typing import Protocol, runtime_checkable


@runtime_checkable
class StatefulModelBuilder(Protocol):
    def __init__(self): ...
