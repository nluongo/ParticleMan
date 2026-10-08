# No scheduler (GPU workstation) specifics for launch.sh (sourced, not executed).

sched_detect() {
    SCHED_NAME=none
    JOB_NUM=""
    WORKDIR=$PWD
    NODES=("$(hostname)")
}

# Whether this shell runs outside the allocation
sched_is_remote() {
    return 1
}

sched_per_node_prefix() {
    echo "ERROR: multi-node launch needs a PBS or Slurm allocation" >&2
    return 1
}

sched_per_gpu_prefix() {
    echo "ERROR: LAUNCHER=$1 needs a PBS or Slurm allocation; use LAUNCHER=torchrun" >&2
    return 1
}
