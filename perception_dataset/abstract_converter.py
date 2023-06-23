from abc import ABCMeta, abstractmethod


class AbstractConverter(object, metaclass=ABCMeta):
    def __init__(
        self,
        input_base: str,
        output_base: str,
    ):
        super().__init__()
        self._input_base = input_base
        self._output_base = output_base

    @abstractmethod
    def convert(self):
        raise NotImplementedError()
