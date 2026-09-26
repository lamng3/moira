"""JSON and DynamoDB serialization for OOT values."""

from __future__ import annotations

import base64
import math
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID

import numpy as np


def _datetime(value: datetime | date) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).replace(
            microsecond=0
        ).isoformat().replace("+00:00", "Z")
    return value.isoformat()


def _bytes(value: bytes | bytearray) -> str:
    try:
        return bytes(value).decode("utf-8")
    except UnicodeDecodeError:
        return "base64:" + base64.b64encode(bytes(value)).decode("ascii")


def _float(value: float) -> float | str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def _custom(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            return str(value)
    return value


def to_dynamodb_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        converted = _float(value)
        return Decimal(str(converted)) if isinstance(converted, float) else converted
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (datetime, date)):
        return _datetime(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return _bytes(value)
    if isinstance(value, Enum):
        return to_dynamodb_safe(value.value)
    if isinstance(value, np.ndarray):
        return [to_dynamodb_safe(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return to_dynamodb_safe(float(value))
    if isinstance(value, np.bool_):
        return bool(value)
    converted = _custom(value)
    if converted is not value:
        return to_dynamodb_safe(converted)
    if isinstance(value, dict):
        return {str(key): to_dynamodb_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_dynamodb_safe(item) for item in value]
    return str(value)


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return _float(value)
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, (datetime, date)):
        return _datetime(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return _bytes(value)
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _float(float(value))
    if isinstance(value, np.bool_):
        return bool(value)
    converted = _custom(value)
    if converted is not value:
        return to_jsonable(converted)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return str(value)


__all__ = ["to_dynamodb_safe", "to_jsonable"]
