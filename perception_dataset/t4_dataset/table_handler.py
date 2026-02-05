from __future__ import annotations

import json
import os.path as osp
from typing import Any, Dict, Generic, List, TypeVar

import attrs
from t4_devkit.common.io import save_json
from t4_devkit.common.serialize import serialize_dataclass, serialize_dataclasses
from t4_devkit.schema import SchemaBase, build_schema
from t4_devkit.schema.tables.registry import SCHEMAS

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

    def get_record_from_token(self, token: str) -> SchemaRecord:
        """Retrieve a record from the table by its token.

        Args:
            token (str): Token of the record to retrieve
        Returns:
            SchemaRecord: The record associated with the given token
        """
        if token not in self._token_to_record:
            raise KeyError(f"Token {token} isn't in table {self._schema_type.__name__}.")
        return self._token_to_record[token]

    def _is_duplicate_record(self, record1: SchemaRecord, record2: SchemaRecord) -> bool:
        """Check if two records are equal excluding the token field.

        Args:
            record1: First record to compare
            record2: Second record to compare

        Returns:
            bool: True if records are duplicates (all fields except token are equal)
        """
        dict1 = serialize_dataclass(record1)
        dict2 = serialize_dataclass(record2)

        # Remove token field from both dictionaries
        dict1.pop("token", None)
        dict2.pop("token", None)

        return dict1 == dict2

    def insert_into_table(self, **kwargs) -> str:
        # Create a temporary record to compare
        temp_record = self._to_record(**kwargs)

        # Check if a record with the same field values (excluding token) already exists
        for existing_token, existing_record in self._token_to_record.items():
            if self._is_duplicate_record(temp_record, existing_record):
                raise ValueError(
                    f"Duplicate record found in table {self._schema_type.__name__}. "
                    f"Existing token: {existing_token}"
                )

        # No duplicate found, add the new record
        self.set_record_to_table(temp_record)
        return temp_record.token

    def update_record_from_token(self, token: str, **kwargs) -> None:
        """Update specific fields of a record in the table.

        Args:
            token (str): Token of the record to update
            **kwargs: Field names and their new values to update

        Raises:
            AssertionError: If token doesn't exist in the table
            AssertionError: If any field name is invalid for the schema type
        """
        assert (
            token in self._token_to_record
        ), f"Token {token} isn't in table {self._schema_type.__name__}."

        # Verify all field names are valid
        for field_name in kwargs:
            assert field_name in self._field_names, (
                f"Field '{field_name}' does not exist in table {self._schema_type.__name__}. "
                f"Available fields: {self._field_names}"
            )

        # Get the current record and update it using attrs.evolve
        current_record = self._token_to_record[token]
        updated_record = attrs.evolve(current_record, **kwargs)
        self._token_to_record[token] = updated_record

    def get_token_from_field(self, field_name: str, field_value: Any) -> str | None:
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
            token
            for token, record in self._token_to_record.items()
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
        return serialize_dataclasses(self.to_records())

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
        save_json(table_data, osp.join(output_dir, f"{self._schema_name}.json"))

    @classmethod
    def from_json(cls, schema_type: SchemaRecord, filepath: str):
        table_handler = cls(schema_type)
        for records in build_schema(get_schema_name(schema_type), filepath):
            table_handler.set_record_to_table(records)
        return table_handler
