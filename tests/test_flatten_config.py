"""Tests for flattening configs into experiment-tracker params."""

from enum import Enum

from omegaconf import OmegaConf

from particleman.loggers import flatten_config


class _Color(Enum):
    RED = "red"


def test_nested_keys_are_dotted() -> None:
    cfg = {"data": {"preprocessing": {"pt_scale": 1000.0}}, "model": {"d_model": 128}}
    assert flatten_config(cfg) == {
        "data.preprocessing.pt_scale": 1000.0,
        "model.d_model": 128,
    }


def test_lists_none_and_enums() -> None:
    cfg = {"a": [1, 2], "b": None, "c": _Color.RED, "d": {}}
    assert flatten_config(cfg) == {"a": "[1, 2]", "b": "None", "c": "red", "d": "{}"}


def test_invalid_key_chars_are_sanitized() -> None:
    cfg = {"data": {"file_event_label_map": {"*ttbar*.root": 1, 11: 2}}}
    assert flatten_config(cfg) == {
        "data.file_event_label_map._ttbar_.root": 1,
        "data.file_event_label_map.11": 2,
    }


def test_omegaconf_container() -> None:
    cfg = OmegaConf.create({"x": {"y": "${z}"}, "z": 3})
    container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(container, dict)
    assert flatten_config(container) == {"x.y": 3, "z": 3}
