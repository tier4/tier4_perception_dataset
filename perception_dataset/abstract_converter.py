from abc import ABCMeta, abstractmethod
from typing import Generic, TypeVar

OUTPUT_TYPE = TypeVar("OUTPUT_TYPE")


class AbstractConverter(Generic[OUTPUT_TYPE], metaclass=ABCMeta):
    def __init__(
        self,
        input_base: str,
        output_base: str,
    ):
        super().__init__()
        self._input_base = input_base
        self._output_base = output_base

    @abstractmethod
    def convert(self) -> OUTPUT_TYPE:
        raise NotImplementedError()
