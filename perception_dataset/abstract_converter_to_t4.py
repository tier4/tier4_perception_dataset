from abc import ABCMeta, abstractmethod
from typing import Tuple


class AbstractAnnotatedToT4Converter(object, metaclass=ABCMeta):
    def __init__(
        self,
        input_base: str,
        output_base: str,
    ) -> None:
        super().__init__()
        self._input_base = input_base
        self._output_base = output_base

    @abstractmethod
    def convert(self) -> Tuple[str, str]:
        raise NotImplementedError()
