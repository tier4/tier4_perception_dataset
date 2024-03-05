from abc import ABCMeta, abstractmethod
import json
import os.path as osp
from typing import Any, Dict, Generic, List, TypeVar

from perception_dataset.utils.gen_tokens import generate_token


class AbstractRecord(metaclass=ABCMeta):
    def __init__(self):
        self._token: str = generate_token(16, "hex")

    @property
    def token(self) -> str:
        return self._token

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError()


T = TypeVar("T", bound=AbstractRecord)


class AbstractTable(Generic[T], metaclass=ABCMeta):
    FILENAME = ""

    def __init__(self):
        self._token_to_record: Dict[str, T] = {}

    def __len__(self) -> int:
        return len(self._token_to_record)

    @abstractmethod
    def _to_record(self, **kwargs) -> T:
        """Return the instance of RecordClass"""
        raise NotImplementedError()

    def set_record_to_table(self, record: T):
        self._token_to_record[record.token] = record

    def insert_into_table(self, **kwargs) -> str:
        record = self._to_record(**kwargs)
        assert isinstance(record, T), "_to_record function must return the instance of RecordClass"
        self.set_record_to_table(record)
        return record.token

    def select_record_from_token(self, token: str) -> T:
        assert (
            token in self._token_to_record
        ), f"Token {token} isn't in table {self.__class__.__name__}."
        return self._token_to_record[token]

    def to_data(self) -> List[Dict[str, Any]]:
        return [rec.to_dict() for rec in self._token_to_record.values()]

    def to_records(self) -> List[T]:
        return list(self._token_to_record.values())

    def to_tokens(self) -> List[str]:
        return list(self._token_to_record.keys())

    def save_json(self, output_dir: str):
        """Save table data to json file

        Args:
            output_dir (str): path to directory
        """
        table_data = self.to_data()
        with open(osp.join(output_dir, self.FILENAME), "w") as f:
            json.dump(table_data, f, indent=4)
