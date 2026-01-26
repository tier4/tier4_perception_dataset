from __future__ import annotations
from t4_devkit.schema import SchemaBase,build_schema
from t4_devkit.common.serialize import serialize_dataclass
from t4_devkit.schema.tables.registry import SCHEMAS
import attrs
import json
import os.path as osp
from typing import Any, Dict, Generic, List, TypeVar

SchemaRecord = TypeVar("SchemaRecord", bound=SchemaBase)


def get_schema_name(schema_type: SchemaRecord) -> str:
    for schema_name, cls in SCHEMAS.schemas.items():
        if cls is schema_type:
            return schema_name
    raise ValueError(f"Schema type {schema_type.__name__} not found in SCHEMAS registry")

class TableHandler(Generic[SchemaRecord]):
    TOKEN_NBYTES = 16

    def __init__(self, schema_type: SchemaRecord):
        self._schema_type = schema_type
        self._schema_name = get_schema_name(schema_type)
        self._token_to_record: Dict[str, SchemaRecord] = {}
        self._field_to_token_cache: Dict[str, Dict[Any, str]] = {}
        self._field_names: set[str] = {f.name for f in attrs.fields(schema_type)}
    @property
    def schema_name(self) -> str:
        return self._schema_name
    
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

    def replace_record_in_table(self, record: SchemaRecord):
        assert (
            record.token in self._token_to_record
        ), f"Token {record.token} isn't in table {self._schema_type.__name__}."
        self._token_to_record[record.token] = record
    def get_token_from_field(self, field_name: str, field_value: Any) -> str:
        """Find token by searching for a unique field value in the table.
        
        Args:
            field_name (str): Name of the field to search by
            field_value (Any): Value of the field to search for
            
        Returns:
            str: Token of the record with matching field value
            
        Raises:
            AssertionError: If field doesn't exist, or if zero or multiple records match the field value
        """
        # Check cache first
        if field_name not in self._field_to_token_cache:
            self._field_to_token_cache[field_name] = {}
        
        if field_value in self._field_to_token_cache[field_name]:
            return self._field_to_token_cache[field_name][field_value]
        
        # Verify field exists on the schema type (check attrs fields)
        assert field_name in self._field_names, (
            f"Field '{field_name}' does not exist in table {self._schema_type.__name__}. "
            f"Available fields: {self._field_names}"
        )
        
        # Search for matching records
        matching_tokens = [
            token for token, record in self._token_to_record.items()
            if getattr(record, field_name) == field_value
        ]
        
        if not matching_tokens:
            return None 
    
        # Assert exactly one match found
        assert len(matching_tokens) == 1, (
            f"Field '{field_name}' with value '{field_value}' must be unique in table "
            f"{self._schema_type.__name__}. Found {len(matching_tokens)} matches."
        )
        
        token = matching_tokens[0]
        # Cache the result
        self._field_to_token_cache[field_name][field_value] = token
        return token

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
        with open(osp.join(output_dir, f"{self._schema_name}.json"), "w") as f:
            json.dump(table_data, f, indent=4)

    @classmethod
    def from_json(cls, schema_type: SchemaRecord, filepath: str):
        table_handler = cls(schema_type)
        for records in build_schema(get_schema_name(schema_type), filepath):    
            table_handler.set_record_to_table(records)
        return table_handler