import abc
from typing import TypeVar

T = TypeVar("T", bound="lmms")


class lmms(abc.ABC):
    def __init__(self) -> None:
        """Defines the interface that should be implemented by all lmms subclasses.
        lmmss are assumed to take image-text as input and yield strings as output
        (inputs/outputs should be tokenization-agnostic.)
        """
        pass
