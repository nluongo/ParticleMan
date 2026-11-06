"""Helpers for turning configs into experiment-tracker parameters."""

import re
from enum import Enum
from collections.abc import Mapping
from typing import Any

# MLflow accepts alphanumerics, underscores, dashes, periods, spaces and slashes.
_INVALID_KEY_CHARS = re.compile(r"[^A-Za-z0-9_\-. /]")


def flatten_config(cfg: Mapping[Any, Any], sep: str = ".") -> dict[str, Any]:
    """Flatten a nested config mapping into a single-level parameter dict.

    Nested mapping keys are joined with ``sep`` (``data.preprocessing.pt_scale``).
    Lists and tuples become their string form, ``None`` becomes ``"None"`` and
    enums become their value. Characters that MLflow rejects in parameter names
    (e.g. the ``*`` in fnmatch patterns) are replaced with ``_``.

    Args:
        cfg: Nested mapping, e.g. the output of ``OmegaConf.to_container``.
        sep: Separator used to join nested keys.

    Returns:
        Flat dict of parameter name to scalar or string value.
    """
    flat: dict[str, Any] = {}

    def _walk(node: Mapping[Any, Any], prefix: str) -> None:
        for key, value in node.items():
            name = _INVALID_KEY_CHARS.sub("_", str(key))
            full_key = f"{prefix}{sep}{name}" if prefix else name
            if isinstance(value, Mapping):
                if value:
                    _walk(value, full_key)
                else:
                    flat[full_key] = "{}"
            elif isinstance(value, Enum):
                flat[full_key] = value.value
            elif isinstance(value, (list, tuple)) or value is None:
                flat[full_key] = str(value)
            else:
                flat[full_key] = value

    _walk(cfg, "")
    return flat
