## Customize Schema

---

You can customize schema classes on your own code, if you need for some reasons.

For example, you can make `Attribute` allow `attribute.json` not to require `description` field as follows:

```python title="custom_attribute.py"
from dataclasses import dataclass
from typing import Any

from typing_extensions import Self

from t4_devkit.schema import SCHEMAS, SchemaName, SchemaBase
from t4_devkit.common.io import load_json


@dataclass
@SCHEMAS.register(SchemaName.ATTRIBUTE, force=True)
class CustomAttribute(SchemaBase):
    """Custom Attribute class ignoring if there is no description field.
    Note that `description` field is mandatory by the original definition.
    """

    token: str
    name: str
    description: str | None

    @classmehod
    def from_json(cls, filepath: str) -> list[Self]:
        objs: list[Self] = []

        record_list: list[dict[str, Any]] = load_json(filepath)
        for record in record_list:
            token: str = record["token"]
            name: str = record["name"]
            # Return None if record does not have description field
            description: str | None = record.get("description")
            objs.append(cls(token=token, name=name, description=description))
        return objs
```
