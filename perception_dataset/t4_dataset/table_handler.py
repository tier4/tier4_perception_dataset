from __future__ import annotations
from t4_devkit.schema import SchemaBase,build_schema
from t4_devkit.common.serialize import serialize_dataclass
from t4_devkit.schema.tables.registry import SCHEMAS
import json
import os.path as osp
from typing import Any, Dict, Generic, List, TypeVar

SchemaRecord = TypeVar("SchemaTable", bound=SchemaBase)


def get_schema_name(schema_type: SchemaRecord) -> str:
    for schema_name, cls in SCHEMAS.schemas.items():
        if cls is schema_type:
            return schema_name
    raise ValueError(f"Schema type {schema_type.__name__} not found in SCHEMAS registry")

class TableHandler:
    TOKEN_NBYTES = 16

    def __init__(self, schema_type: SchemaRecord):
        self._schema_type = schema_type
        self._token_to_record: Dict[str, SchemaRecord] = {}
        
    def __len__(self) -> int:
        return len(self._token_to_record)

    def _to_record(self, **kwargs) -> SchemaRecord:
       return self._schema_type.new(kwargs, token_nbytes=self.TOKEN_NBYTES)

    def set_record_to_table(self, record: SchemaRecord):
        self._token_to_record[record.token] = record

    def insert_into_table(self, **kwargs) -> str:
        record = self._to_record(**kwargs)
        self.set_record_to_table(record)
        return record.token

    def select_record_from_token(self, token: str) -> SchemaRecord:
        assert (
            token in self._token_to_record
        ), f"Token {token} isn't in table {self._schema_type.__name__}."
        return self._token_to_record[token]

    def to_data(self) -> List[Dict[str, Any]]:
        return [serialize_dataclass(rec) for rec in self._token_to_record.values()]

    def to_records(self) -> List[SchemaRecord]:
        return list(self._token_to_record.values())

    def to_tokens(self) -> List[str]:
        return list(self._token_to_record.keys())

    def save_json(self, output_dir: str):
        """Save table data to json file

        Args:
            output_dir (str): path to directory
        """
        table_data = self.to_data()
        with open(osp.join(output_dir, f"{get_schema_name(self._schema_type)}.json"), "w") as f:
            json.dump(table_data, f, indent=4)

    @classmethod
    def from_json(cls, schema_type: SchemaRecord, filepath: str):
        table_handler = cls(schema_type)
        for records in build_schema(get_schema_name(schema_type), filepath):    
            table_handler.set_record_to_table(records)
        return table_handler