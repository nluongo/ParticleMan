"""Environment contract between scripts/run.py and scripts/jobs/launch.sh."""

import shlex
from typing import Any, Dict, Mapping, Optional, Sequence


def build_job_env(
    cluster: Mapping[str, Any],
    resources: Mapping[str, Any],
    job: Mapping[str, Any],
    output: Mapping[str, Any],
    repo_root: str,
    train_script: str,
    hydra_args: Sequence[str],
    timestamp: str,
) -> Dict[str, str]:
    """
    Build the environment variables the job script expects.

    Args:
        cluster: Site config (launcher, setup_script, master_port, env)
        resources: Job size config (gpus_per_node, omp_num_threads); a null
            gpus_per_node leaves GPUS_PER_NODE unset
        job: Job config (name, nccl_debug)
        output: Composed training ``output`` config (output_dir,
            experiment_name, run_name), used to locate the run directory
        repo_root: Absolute path of the repository
        train_script: Absolute path of the Python entry point
        hydra_args: Hydra overrides for the entry point
        timestamp: Run timestamp shared with OutputManager

    Returns:
        Mapping of variable name to string value
    """
    env: Dict[str, str] = {
        "REPO_ROOT": repo_root,
        "TRAIN_SCRIPT": train_script,
        # shlex-quoted; launch.sh restores the argument list with eval
        "HYDRA_ARGS": shlex.join(hydra_args),
        "OMP_NUM_THREADS": str(resources.get("omp_num_threads") or 8),
        "MASTER_PORT": str(cluster.get("master_port") or 29500),
        "LAUNCHER": str(cluster.get("launcher") or "torchrun"),
        "NCCL_DEBUG": str(job.get("nccl_debug") or "WARN"),
        "JOB_NAME": str(job["name"]),
        "OUTPUT_DIR": str(output.get("output_dir") or "outputs"),
        "EXPERIMENT_NAME": str(output["experiment_name"]),
        "TIMESTAMP": timestamp,
    }
    gpus_per_node = resources.get("gpus_per_node")
    if gpus_per_node is not None:
        # Unset: launch.sh counts the node's GPUs with nvidia-smi
        env["GPUS_PER_NODE"] = str(gpus_per_node)
    setup_script: Optional[str] = cluster.get("setup_script")
    if setup_script:
        env["SETUP_SCRIPT"] = str(setup_script)
    run_name: Optional[str] = output.get("run_name")
    if run_name:
        env["RUN_NAME"] = str(run_name)
    for key, value in (cluster.get("env") or {}).items():
        env[str(key)] = str(value)
    return env
