"""Tests for scheduler submit commands and the job environment contract."""

import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pytest
from omegaconf import OmegaConf

from particleman.config import Launcher
from particleman.scheduling import (
    PBSScheduler,
    SlurmScheduler,
    build_job_env,
    check_launcher,
    get_scheduler,
    submit_job,
)

PBS_CLUSTER: Dict[str, Any] = {
    "scheduler": "pbs",
    "account": "ATLAS-HEP-GROUP",
    "queue": "backfill",
    "filesystems": None,
    "launcher": "torchrun",
    "setup_script": "my_setup.sh",
    "master_port": 29500,
    "env": {"NCCL_NET_GDR_LEVEL": 0, "TMPDIR": "/tmp"},
    "extra_args": [],
}
SLURM_CLUSTER: Dict[str, Any] = {
    "scheduler": "slurm",
    "account": "my-account",
    "queue": "gpu",
    "qos": None,
    "launcher": "srun",
    "setup_script": None,
    "master_port": 29500,
    "env": {},
    "extra_args": ["--constraint=a100"],
}
MULTI_NODE: Dict[str, Any] = {
    "nnodes": 2,
    "gpus_per_node": 4,
    "cpus_per_node": 16,
    "mem": "64gb",
    "walltime": "12:00:00",
    "omp_num_threads": 8,
    "queue": "regular",
}
SINGLE_GPU: Dict[str, Any] = {
    "nnodes": 1,
    "gpus_per_node": 1,
    "cpus_per_node": 8,
    "mem": "32gb",
    "walltime": "4:00:00",
    "omp_num_threads": 8,
    "queue": None,
}
JOB: Dict[str, Any] = {"name": "particleman_train", "nccl_debug": "WARN"}


def test_get_scheduler() -> None:
    assert isinstance(get_scheduler("pbs"), PBSScheduler)
    assert isinstance(get_scheduler("slurm"), SlurmScheduler)
    with pytest.raises(ValueError):
        get_scheduler("lsf")


@pytest.mark.parametrize(
    "scheduler,launcher,ok",
    [
        ("pbs", "torchrun", True),
        ("pbs", "mpiexec", True),
        ("pbs", "srun", False),
        ("slurm", "torchrun", True),
        ("slurm", "srun", True),
        ("slurm", "mpiexec", False),
        ("none", "torchrun", True),
        ("none", "srun", False),
    ],
)
def test_check_launcher(scheduler: str, launcher: str, ok: bool) -> None:
    if ok:
        check_launcher(scheduler, launcher)
    else:
        with pytest.raises(ValueError):
            check_launcher(scheduler, launcher)


def test_pbs_command_multi_node() -> None:
    cmd = PBSScheduler().build_submit_command(
        PBS_CLUSTER, MULTI_NODE, JOB, "launch.sh", ["A", "B"]
    )
    assert cmd == [
        "qsub",
        "-A", "ATLAS-HEP-GROUP",
        "-N", "particleman_train",
        "-q", "regular",  # resources.queue overrides cluster.queue
        "-l", "walltime=12:00:00",
        "-l", "select=2:ncpus=16:ngpus=4:mem=64gb",
        "-j", "oe",
        "-v", "A,B",
        "launch.sh",
    ]


def test_pbs_omits_null_resources_and_adds_filesystems() -> None:
    cluster = {**PBS_CLUSTER, "filesystems": "home:eagle"}
    resources = {**SINGLE_GPU, "cpus_per_node": None, "mem": None}
    cmd = PBSScheduler().build_submit_command(cluster, resources, JOB, "launch.sh", [])
    assert "select=1:ngpus=1" in cmd
    assert "filesystems=home:eagle" in cmd
    assert "-v" not in cmd
    assert cmd[cmd.index("-q") + 1] == "backfill"


def test_slurm_command_multi_node() -> None:
    cmd = SlurmScheduler().build_submit_command(
        SLURM_CLUSTER, MULTI_NODE, JOB, "launch.sh", ["A"]
    )
    assert cmd == [
        "sbatch",
        "--parsable",
        "--account=my-account",
        "--job-name=particleman_train",
        "--partition=regular",
        "--time=12:00:00",
        "--nodes=2",
        "--ntasks-per-node=4",
        "--gpus-per-node=4",
        "--cpus-per-task=4",
        "--mem=64G",
        "--output=%x.o%j",
        "--export=ALL",
        "--constraint=a100",
        "launch.sh",
    ]



def test_slurm_command_gpu_type() -> None:
    resources = {**SINGLE_GPU, "gpu_type": "rtx_pro_6000_blackwell"}
    cmd = SlurmScheduler().build_submit_command(
        SLURM_CLUSTER, resources, JOB, "launch.sh", []
    )
    assert "--gres=gpu:rtx_pro_6000_blackwell:1" in cmd
    assert not any(a.startswith("--gpus-per-node") for a in cmd)

@pytest.mark.parametrize("mem,expected", [("64gb", "64G"), ("32GB", "32G"), ("500M", "500M"), ("4096", "4096")])
def test_slurm_mem_string(mem: str, expected: str) -> None:
    assert SlurmScheduler.mem_string(mem) == expected


def test_parse_job_id() -> None:
    assert PBSScheduler().parse_job_id("12345.polaris-pbs-01.hsn\n") == "12345"
    assert SlurmScheduler().parse_job_id("6789\n") == "6789"
    assert SlurmScheduler().parse_job_id("6789;cluster\n") == "6789"


def test_build_job_env() -> None:
    env = build_job_env(
        cluster=PBS_CLUSTER,
        resources=MULTI_NODE,
        job=JOB,
        output={"output_dir": "outputs", "experiment_name": "Exp", "run_name": None},
        repo_root="/repo",
        train_script="/repo/scripts/train.py",
        hydra_args=["data=bbllv08", "output.experiment_name=a b"],
        timestamp="1700000000",
    )
    assert shlex.split(env["HYDRA_ARGS"]) == ["data=bbllv08", "output.experiment_name=a b"]
    assert env["GPUS_PER_NODE"] == "4"
    assert env["LAUNCHER"] == "torchrun"
    assert env["SETUP_SCRIPT"] == "my_setup.sh"
    assert env["NCCL_NET_GDR_LEVEL"] == "0"
    assert env["EXPERIMENT_NAME"] == "Exp"
    assert "RUN_NAME" not in env
    assert all(isinstance(v, str) for v in env.values())


def test_build_job_env_without_gpus_per_node() -> None:
    env = build_job_env(
        cluster=PBS_CLUSTER,
        resources={**SINGLE_GPU, "gpus_per_node": None},
        job=JOB,
        output={"experiment_name": "Exp"},
        repo_root="/repo",
        train_script="/repo/scripts/train.py",
        hydra_args=[],
        timestamp="1",
    )
    # launch.sh counts the GPUs itself
    assert "GPUS_PER_NODE" not in env


def _submit_cfg(tmp_path: Path, **resources: Any) -> Any:
    return OmegaConf.create(
        {
            # Enum, as composed from the ClusterConfig schema
            "cluster": {**SLURM_CLUSTER, "launcher": Launcher.srun},
            "resources": {**SINGLE_GPU, **resources},
            "job": {**JOB, "mode": "submit", "dry_run": False},
            "output": {
                "output_dir": str(tmp_path / "outputs"),
                "experiment_name": "Exp",
                "run_name": None,
            },
        }
    )


def test_submit_job_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("dry run must not submit")

    monkeypatch.setattr(subprocess, "run", fail)
    job_id = submit_job(
        _submit_cfg(tmp_path), ["data=x"], "17", repo_root=tmp_path, dry_run=True
    )
    assert job_id is None
    assert not (tmp_path / "outputs").exists()


def test_submit_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[Dict[str, Any]] = []

    def fake_run(cmd: List[str], **kwargs: Any) -> subprocess.CompletedProcess:
        calls.append({"cmd": cmd, "env": kwargs["env"]})
        return subprocess.CompletedProcess(cmd, 0, stdout="4242\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    job_id = submit_job(
        _submit_cfg(tmp_path),
        ["data=x", "output.timestamp=17"],
        "17",
        repo_root=tmp_path,
    )
    assert job_id == "4242"
    (call,) = calls
    assert call["cmd"][0] == "sbatch"
    assert call["cmd"][-1] == str(tmp_path / "scripts" / "jobs" / "launch.sh")
    assert call["env"]["LAUNCHER"] == "srun"
    assert call["env"]["TRAIN_SCRIPT"] == str(tmp_path / "scripts" / "train.py")
    assert shlex.split(call["env"]["HYDRA_ARGS"]) == ["data=x", "output.timestamp=17"]
    assert (tmp_path / "outputs" / "Exp" / "17_slurm4242" / "submit_command.sh").exists()


def test_submit_job_needs_gpus_per_node(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="gpus_per_node"):
        submit_job(
            _submit_cfg(tmp_path, gpus_per_node=None), [], "1", tmp_path, dry_run=True
        )
