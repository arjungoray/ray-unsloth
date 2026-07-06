"""JSON Schema generation for ray-unsloth runtime configs."""

from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Annotated, Any, Union, get_args, get_origin, get_type_hints

from ray_unsloth.config import RuntimeConfig

JSONSchema = dict[str, Any] | bool

__all__ = ["config_json_schema"]


def config_json_schema() -> dict[str, Any]:
    schema = _schema_for_dataclass(RuntimeConfig)
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "ray-unsloth RuntimeConfig"
    return schema


def _schema_for_dataclass(cls: type[Any]) -> dict[str, Any]:
    hints = get_type_hints(cls, include_extras=True)
    properties: dict[str, JSONSchema] = {}
    required: list[str] = []
    for field in fields(cls):
        field_type = hints.get(field.name, Any)
        schema = _schema_for_type(field_type)
        if field.default is MISSING and field.default_factory is MISSING:
            required.append(field.name)
        else:
            schema = _attach_default(schema, _default_value(field))
        properties[field.name] = schema
    result: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
    }
    if required:
        result["required"] = required
    return result


def _schema_for_type(tp: Any) -> JSONSchema:
    if tp is Any or tp is object:
        return True
    origin = get_origin(tp)
    if origin is Annotated:
        return _schema_for_type(get_args(tp)[0])
    if origin in (list, tuple):
        args = get_args(tp)
        item_type = args[0] if args else Any
        return {"type": "array", "items": _schema_for_type(item_type)}
    if origin in (dict,):
        args = get_args(tp)
        value_type = args[1] if len(args) > 1 else Any
        return {"type": "object", "additionalProperties": _schema_for_type(value_type)}
    if origin in (UnionType, Union):
        return _schema_for_union(get_args(tp))
    if isinstance(tp, type):
        if is_dataclass(tp):
            return _schema_for_dataclass(tp)
        if issubclass(tp, bool):
            return {"type": "boolean"}
        if issubclass(tp, int):
            return {"type": "integer"}
        if issubclass(tp, float):
            return {"type": "number"}
        if issubclass(tp, str):
            return {"type": "string"}
        if issubclass(tp, Path):
            return {"type": "string"}
        if issubclass(tp, dict):
            return {"type": "object", "additionalProperties": True}
        if issubclass(tp, list):
            return {"type": "array", "items": True}
    return True


def _schema_for_union(args: tuple[Any, ...]) -> JSONSchema:
    if not args:
        return True
    schemas = [_schema_for_type(arg) for arg in args]
    if any(schema is True for schema in schemas):
        return True
    if len(schemas) == 1:
        return schemas[0]
    return {"anyOf": [schema for schema in schemas]}


def _attach_default(schema: JSONSchema, default: Any) -> JSONSchema:
    if schema is True:
        return {"default": default}
    result = dict(schema)
    result["default"] = default
    return result


def _default_value(field) -> Any:
    if field.default is not MISSING:
        return _to_jsonable(field.default)
    if field.default_factory is not MISSING:  # type: ignore[truthy-function]
        return _to_jsonable(field.default_factory())
    return None


def _to_jsonable(value: Any) -> Any:
    from dataclasses import asdict, is_dataclass

    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
