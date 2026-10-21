"""Batch scheduler backends for job submission.

Each scheduler turns the composed ``cluster`` / ``resources`` / ``job`` submit
configs into a submission command. Job settings are passed to the job script
as environment variables *by name*: the caller sets them in the submitting
process's environment and the scheduler copies them into the job, which avoids
quoting problems with values containing commas or spaces.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Sequence, Type

# Launchers each scheduler can run; torchrun works everywhere.
LAUNCHERS_BY_SCHEDULER: Dict[str, Sequence[str]] = {
    "none": ("torchrun",),
    "pbs": ("torchrun", "mpiexec"),
    "slurm": ("torchrun", "srun"),
}


class Scheduler(ABC):
    """Base class for batch schedulers."""

    #: Scheduler name; also the run-name suffix used by OutputManager.
    name: str = ""

    @abstractmethod
    def build_submit_command(
        self,
        cluster: Mapping[str, Any],
        resources: Mapping[str, Any],
        job: Mapping[str, Any],
        script: str,
        env_names: Sequence[str],
    ) -> List[str]:
        """
        Build the submission command.

        Args:
            cluster: Site config (account, queue, extra_args, ...)
            resources: Job size config (nnodes, gpus_per_node, ...)
            job: Job config (name, ...)
            script: Path to the job script
            env_names: Names of env vars to forward into the job

        Returns:
            Command as a list of arguments
        """

    @abstractmethod
    def parse_job_id(self, stdout: str) -> str:
        """Extract the numeric job ID from the submission command's stdout."""

    @staticmethod
    def queue(cluster: Mapping[str, Any], resources: Mapping[str, Any]) -> Optional[str]:
        """Resources may override the cluster's default queue/partition."""
        return resources.get("queue") or cluster.get("queue")


class PBSScheduler(Scheduler):
    """PBS Pro (qsub)."""

    name = "pbs"

    @staticmethod
    def select_string(resources: Mapping[str, Any]) -> str:
        """Build the ``select=`` chunk spec, omitting unset fields."""
        parts = [f"select={resources.get('nnodes', 1)}"]
        for key, pbs_key in (
            ("cpus_per_node", "ncpus"),
            ("gpus_per_node", "ngpus"),
            ("mem", "mem"),
        ):
            value = resources.get(key)
            if value:
                parts.append(f"{pbs_key}={value}")
        return ":".join(parts)

    def build_submit_command(
        self,
        cluster: Mapping[str, Any],
        resources: Mapping[str, Any],
        job: Mapping[str, Any],
        script: str,
        env_names: Sequence[str],
    ) -> List[str]:
        cmd = ["qsub"]
        if cluster.get("account"):
            cmd += ["-A", str(cluster["account"])]
        cmd += ["-N", str(job["name"])]
        queue = self.queue(cluster, resources)
        if queue:
            cmd += ["-q", str(queue)]
        if resources.get("walltime"):
            cmd += ["-l", f"walltime={resources['walltime']}"]
        cmd += ["-l", self.select_string(resources)]
        if cluster.get("filesystems"):
            cmd += ["-l", f"filesystems={cluster['filesystems']}"]
        cmd += ["-j", "oe"]
        if env_names:
            # Bare names: qsub takes each value from its own environment
            cmd += ["-v", ",".join(env_names)]
        cmd += [str(a) for a in cluster.get("extra_args") or []]
        cmd.append(script)
        return cmd

    def parse_job_id(self, stdout: str) -> str:
        # e.g. "12345.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov"
        return stdout.strip().split(".")[0]


class SlurmScheduler(Scheduler):
    """Slurm (sbatch)."""

    name = "slurm"

    @staticmethod
    def mem_string(mem: str) -> str:
        """Convert PBS-style memory (e.g. '64gb') to Slurm style ('64G')."""
        value = str(mem).strip()
        if value[-1:] in ("b", "B") and value[-2:-1].isalpha():
            value = value[:-1]
        return value.upper()

    def build_submit_command(
        self,
        cluster: Mapping[str, Any],
        resources: Mapping[str, Any],
        job: Mapping[str, Any],
        script: str,
        env_names: Sequence[str],
    ) -> List[str]:
        cmd = ["sbatch", "--parsable"]
        if cluster.get("account"):
            cmd.append(f"--account={cluster['account']}")
        cmd.append(f"--job-name={job['name']}")
        queue = self.queue(cluster, resources)
        if queue:
            cmd.append(f"--partition={queue}")
        if cluster.get("qos"):
            cmd.append(f"--qos={cluster['qos']}")
        if resources.get("walltime"):
            cmd.append(f"--time={resources['walltime']}")

        gpus = int(resources.get("gpus_per_node") or 0)
        cmd.append(f"--nodes={resources.get('nnodes', 1)}")
        # One task per GPU so srun launches can fill the allocation
        cmd.append(f"--ntasks-per-node={max(gpus, 1)}")
        if gpus:
            cmd.append(f"--gpus-per-node={gpus}")
        cpus = resources.get("cpus_per_node")
        if cpus:
            cmd.append(f"--cpus-per-task={max(int(cpus) // max(gpus, 1), 1)}")
        if resources.get("mem"):
            cmd.append(f"--mem={self.mem_string(resources['mem'])}")

        # Same <job_name>.o<job_id> file name as PBS -j oe
        cmd.append("--output=%x.o%j")
        # Forward the submitting environment (which holds env_names)
        cmd.append("--export=ALL")
        cmd += [str(a) for a in cluster.get("extra_args") or []]
        cmd.append(script)
        return cmd

    def parse_job_id(self, stdout: str) -> str:
        # --parsable prints "<job_id>" or "<job_id>;<cluster>"
        return stdout.strip().split(";")[0]


_SCHEDULERS: Dict[str, Type[Scheduler]] = {
    PBSScheduler.name: PBSScheduler,
    SlurmScheduler.name: SlurmScheduler,
}


def get_scheduler(name: str) -> Scheduler:
    """Return the scheduler backend for ``name`` ("pbs" or "slurm")."""
    try:
        return _SCHEDULERS[name]()
    except KeyError:
        raise ValueError(
            f"Unknown scheduler {name!r}; expected one of {sorted(_SCHEDULERS)}"
        ) from None


def check_launcher(scheduler: str, launcher: str) -> None:
    """
    Raise ValueError if ``launcher`` can't be used under ``scheduler``.

    srun only exists under Slurm; the mpiexec launch path targets PBS (PALS/MPICH).
    Without a scheduler ("none") only torchrun is available.
    """
    allowed = LAUNCHERS_BY_SCHEDULER.get(scheduler, ())
    if launcher not in allowed:
        raise ValueError(
            f"launcher={launcher!r} is not supported with scheduler={scheduler!r}; "
            f"use one of {list(allowed)}"
        )
