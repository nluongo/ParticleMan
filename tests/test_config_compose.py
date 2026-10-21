"""Every root config composes with every group option against the schemas."""

from pathlib import Path
from typing import Iterator, List

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from particleman.config import register_configs

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
ROOTS = ["config", "bbllv08", "mc20_ttbar"]
GROUPS = [
    "data",
    "model",
    "training",
    "output",
    "distributed",
    "cluster",
    "resources",
    "job",
]


@pytest.fixture(scope="module", autouse=True)
def hydra_context() -> Iterator[None]:
    register_configs()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        yield


def _options() -> List[str]:
    return [
        f"{group}={path.stem}"
        for group in GROUPS
        for path in sorted((CONFIG_DIR / group).glob("*.yaml"))
    ]


def _compose(root: str, overrides: List[str]) -> DictConfig:
    return compose(config_name=root, overrides=overrides)


@pytest.mark.parametrize("root", ROOTS)
@pytest.mark.parametrize("option", _options())
def test_root_composes_with_group_option(root: str, option: str) -> None:
    _compose(root, [option])


def test_default_runs_locally() -> None:
    cfg = _compose("config", [])
    assert cfg.cluster.scheduler == "none"
    assert cfg.cluster.launcher == "torchrun"
    assert cfg.resources.gpus_per_node is None
    assert cfg.job.mode == "auto"
    # Values come from the YAML files, not the dataclass defaults
    assert list(cfg.data.source.files) == ["data/particles.h5"]


def test_resources_accept_null() -> None:
    cfg = _compose(
        "config",
        ["cluster=swing", "resources.cpus_per_node=null", "resources.mem=null"],
    )
    assert cfg.resources.cpus_per_node is None
    assert cfg.resources.mem is None


@pytest.mark.parametrize(
    "override", ["cluster.launcher=bogus", "job.unknown=1", "resources.nnodes=two"]
)
def test_schema_rejects_bad_values(override: str) -> None:
    with pytest.raises(Exception):
        _compose("config", [override])
