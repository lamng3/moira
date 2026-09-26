from __future__ import annotations

import json
import re
from typing import List, Union


def coerce_terms(term: Union[str, List[str]]) -> List[str]:
    """Normalize a term or a delimited/JSON-encoded term list."""
    if isinstance(term, list):
        return [str(value).strip() for value in term if str(value).strip()]
    if isinstance(term, str):
        value = term.strip()
        if value.startswith("[") and value.endswith("]"):
            try:
                decoded = json.loads(value)
                if isinstance(decoded, list):
                    return [
                        str(item).strip()
                        for item in decoded
                        if str(item).strip()
                    ]
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        terms = [
            part.strip()
            for part in re.split(r"[|,;\n]+", value)
            if part.strip()
        ]
        return terms or [value]
    return [str(term).strip()]
