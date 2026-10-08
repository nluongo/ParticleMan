# PBS Pro specifics for launch.sh (sourced, not executed).

sched_detect() {
    SCHED_NAME=pbs
    JOB_NUM=${PBS_JOBID%%.*}
    WORKDIR=${PBS_O_WORKDIR:-$PWD}
    if [ -n "${PBS_NODEFILE}" ] && [ -f "${PBS_NODEFILE}" ]; then
        # PBS_NODEFILE may list a host once per chunk/CPU
        NODES=()
        while read -r node; do
            [ -n "${node}" ] && NODES+=("${node}")
        done < <(sort -u "${PBS_NODEFILE}")
    else
        NODES=("$(hostname)")
    fi
    # Unique-host file for mpiexec
    HOSTFILE=$(mktemp)
    printf '%s\n' "${NODES[@]}" > "${HOSTFILE}"
}

# Whether this shell runs outside the allocation
sched_is_remote() {
    return 1  # the job script and qsub -I shells run on the first allocated node
}

# One process per node (used to start torchrun on every node)
sched_per_node_prefix() {
    PREFIX=(mpiexec -n "${NNODES}" --ppn 1 --hostfile "${HOSTFILE}" --cpu-bind none)
}

# One rank per GPU
sched_per_gpu_prefix() {
    if [ "$1" != "mpiexec" ]; then
        echo "ERROR: LAUNCHER=$1 is not available under PBS (use torchrun or mpiexec)" >&2
        return 1
    fi
    PREFIX=(
        mpiexec -n "${TOTAL_GPUS}" --ppn "${GPUS_PER_NODE}"
        --hostfile "${HOSTFILE}" --cpu-bind none
    )
}
