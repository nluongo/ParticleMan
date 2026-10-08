# Slurm specifics for launch.sh (sourced, not executed).

sched_detect() {
    SCHED_NAME=slurm
    JOB_NUM=${SLURM_JOB_ID}
    WORKDIR=${SLURM_SUBMIT_DIR:-$PWD}
    if [ -n "${SLURM_JOB_NODELIST}" ]; then
        NODES=()
        while read -r node; do
            [ -n "${node}" ] && NODES+=("${node}")
        done < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
    else
        NODES=("$(hostname)")
    fi
}

# Whether this shell runs outside the allocation
sched_is_remote() {
    # SLURMD_NODENAME is set in batch scripts and job steps, not in a salloc
    # shell that stays on the login node
    [ -z "${SLURMD_NODENAME}" ]
}

# One process per node (used to start torchrun on every node). Steps don't
# inherit GPU/CPU counts reliably, so request them explicitly.
sched_per_node_prefix() {
    PREFIX=(
        srun --nodes="${NNODES}" --ntasks-per-node=1
        --gpus-per-node="${GPUS_PER_NODE}" --cpu-bind=none
    )
    if [ -n "${SLURM_CPUS_ON_NODE}" ]; then
        PREFIX+=(--cpus-per-task="${SLURM_CPUS_ON_NODE}")
    fi
}

# One rank per GPU; every rank sees all node GPUs and picks one by local rank
sched_per_gpu_prefix() {
    if [ "$1" != "srun" ]; then
        echo "ERROR: LAUNCHER=$1 is not available under Slurm (use torchrun or srun)" >&2
        return 1
    fi
    PREFIX=(
        srun --nodes="${NNODES}" --ntasks-per-node="${GPUS_PER_NODE}"
        --gpus-per-node="${GPUS_PER_NODE}" --cpu-bind=none
    )
    if [ -n "${SLURM_CPUS_PER_TASK}" ]; then
        PREFIX+=(--cpus-per-task="${SLURM_CPUS_PER_TASK}")
    fi
}
