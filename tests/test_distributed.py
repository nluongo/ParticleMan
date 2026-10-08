"""Tests for launcher environment detection in distributed setup."""

import pytest

from particleman.training.distributed import detect_rank_info


def test_no_launcher_is_single_process() -> None:
    """With no launcher env vars, run as a single process."""
    assert detect_rank_info({}) == (0, 1, 0, "none")


def test_torchrun() -> None:
    """torchrun sets RANK/WORLD_SIZE/LOCAL_RANK."""
    env = {"RANK": "5", "WORLD_SIZE": "8", "LOCAL_RANK": "1"}
    assert detect_rank_info(env) == (5, 8, 1, "torchrun")


def test_slurm_srun() -> None:
    """srun sets SLURM_PROCID/SLURM_NTASKS/SLURM_LOCALID."""
    env = {"SLURM_PROCID": "6", "SLURM_NTASKS": "8", "SLURM_LOCALID": "2"}
    assert detect_rank_info(env) == (6, 8, 2, "slurm")


@pytest.mark.parametrize("local_var", ["PALS_LOCAL_RANKID", "MPI_LOCALRANKID"])
def test_mpich_mpiexec(local_var: str) -> None:
    """MPICH/PALS mpiexec sets PMI_RANK/PMI_SIZE and a local rank variable."""
    env = {"PMI_RANK": "3", "PMI_SIZE": "4", local_var: "3"}
    assert detect_rank_info(env) == (3, 4, 3, "mpich")


def test_openmpi() -> None:
    """OpenMPI mpirun sets OMPI_COMM_WORLD_* variables."""
    env = {
        "OMPI_COMM_WORLD_RANK": "1",
        "OMPI_COMM_WORLD_SIZE": "2",
        "OMPI_COMM_WORLD_LOCAL_RANK": "1",
    }
    assert detect_rank_info(env) == (1, 2, 1, "openmpi")


def test_torchrun_inside_srun_prefers_torchrun() -> None:
    """torchrun launched per node by srun must use torchrun's per-GPU ranks."""
    env = {
        "SLURM_PROCID": "1",
        "SLURM_NTASKS": "2",
        "SLURM_LOCALID": "0",
        "RANK": "7",
        "WORLD_SIZE": "8",
        "LOCAL_RANK": "3",
    }
    assert detect_rank_info(env) == (7, 8, 3, "torchrun")


def test_local_rank_derived_from_gpus_per_node_argument() -> None:
    """Without a local rank variable, local_rank is rank modulo GPUs per node."""
    env = {"PMI_RANK": "6", "PMI_SIZE": "8"}
    assert detect_rank_info(env, gpus_per_node=4) == (6, 8, 2, "mpich")


def test_local_rank_derived_from_gpus_per_node_env() -> None:
    """GPUS_PER_NODE (exported by the job script) is used when no argument is given."""
    env = {"PMI_RANK": "5", "PMI_SIZE": "8", "GPUS_PER_NODE": "4"}
    assert detect_rank_info(env) == (5, 8, 1, "mpich")


def test_incomplete_torchrun_vars_fall_through() -> None:
    """RANK without WORLD_SIZE is not treated as a torchrun launch."""
    env = {"RANK": "2", "SLURM_PROCID": "1", "SLURM_NTASKS": "2", "SLURM_LOCALID": "1"}
    assert detect_rank_info(env) == (1, 2, 1, "slurm")
