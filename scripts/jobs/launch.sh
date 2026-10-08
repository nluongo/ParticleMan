#!/bin/bash
# ============================================================================
# ParticleMan job launcher (PBS, Slurm, or no scheduler)
# ============================================================================
#
# Starts single-GPU, multi-GPU or multi-node training. Scheduler-specific
# pieces (node list, job ID, per-node/per-GPU process starters) live in
# scheduler_<name>.sh next to this file; everything else is shared.
#
# Batch mode: submitted by scripts/submit.py, which passes all settings as
# environment variables (TRAIN_SCRIPT, HYDRA_ARGS, LAUNCHER, GPUS_PER_NODE, ...).
#
# Interactive mode: inside an allocation (qsub -I / salloc) or on a GPU
# workstation, pass the entry point and Hydra overrides as arguments:
#
#   bash scripts/jobs/launch.sh scripts/train.py data=bbllv08 training=quick_test
#   LAUNCHER=srun bash scripts/jobs/launch.sh scripts/train.py ...
#
# Launchers (LAUNCHER, default torchrun):
#   torchrun  one torchrun per node, one worker per GPU (any scheduler)
#   mpiexec   one rank per GPU via PBS mpiexec (PALS/MPICH)
#   srun      one rank per GPU via Slurm srun
#
# Batch output: outputs/<experiment>/<timestamp>_<scheduler><job_id>/ gets
# launch_command.sh and job_output.log in addition to OutputManager's files.
# ============================================================================

set -o pipefail

# sbatch runs a spooled copy of this script, so prefer REPO_ROOT from submit.py
if [ -z "${REPO_ROOT}" ]; then
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
JOBS_DIR="${REPO_ROOT}/scripts/jobs"

# ============================================================================
# Entry point and arguments
# ============================================================================

if [ $# -gt 0 ]; then
    BATCH=0
    TRAIN_SCRIPT="$1"
    shift
    TRAIN_ARGS=("$@")
elif [ -n "${TRAIN_SCRIPT}" ]; then
    BATCH=1
    # HYDRA_ARGS is shell-quoted by submit.py (shlex.join)
    eval "TRAIN_ARGS=(${HYDRA_ARGS})"
else
    echo "Usage: $0 <script.py> [hydra overrides...]" >&2
    echo "  (or set TRAIN_SCRIPT/HYDRA_ARGS, as scripts/submit.py does)" >&2
    exit 1
fi

# ============================================================================
# Scheduler detection
# ============================================================================

if [ -n "${PBS_JOBID}" ]; then
    source "${JOBS_DIR}/scheduler_pbs.sh"
elif [ -n "${SLURM_JOB_ID}" ]; then
    source "${JOBS_DIR}/scheduler_slurm.sh"
else
    source "${JOBS_DIR}/scheduler_none.sh"
fi

# Sets SCHED_NAME, JOB_NUM, WORKDIR and NODES (unique hostnames)
sched_detect

echo "=============================================="
echo "Scheduler: ${SCHED_NAME}  Job ID: ${JOB_NUM:-none}"
echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "=============================================="

if [ "${BATCH}" -eq 1 ]; then
    cd "${WORKDIR}" || exit 1
fi

if [ -n "${SETUP_SCRIPT}" ]; then
    if [ -f "${SETUP_SCRIPT}" ]; then
        echo "Sourcing setup script: ${SETUP_SCRIPT}"
        # shellcheck disable=SC1090
        source "${SETUP_SCRIPT}"
    else
        echo "WARNING: setup script not found: ${SETUP_SCRIPT}" >&2
    fi
fi

# ============================================================================
# Topology and environment
# ============================================================================

NNODES=${#NODES[@]}
if [ -z "${GPUS_PER_NODE}" ]; then
    GPUS_PER_NODE=$(( $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l) ))
    [ "${GPUS_PER_NODE}" -ge 1 ] 2>/dev/null || GPUS_PER_NODE=1
fi
TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))
LAUNCHER=${LAUNCHER:-torchrun}

export GPUS_PER_NODE
export MASTER_ADDR=${MASTER_ADDR:-${NODES[0]}}
export MASTER_PORT=${MASTER_PORT:-29500}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

echo ""
echo "Topology:"
echo "  Nodes: ${NNODES} (${NODES[*]})"
echo "  GPUs per node: ${GPUS_PER_NODE}"
echo "  Total GPUs: ${TOTAL_GPUS}"
echo "  Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Launcher: ${LAUNCHER}"
echo "  Entry point: ${TRAIN_SCRIPT} ${TRAIN_ARGS[*]}"
echo ""

# ============================================================================
# Build launch command
# ============================================================================

# True when this shell isn't on an allocated node (e.g. a Slurm salloc shell
# on the login node), so even single-node work must go through the scheduler
REMOTE=0
sched_is_remote && REMOTE=1

if [ "${TOTAL_GPUS}" -le 1 ]; then
    PREFIX=()
    if [ "${REMOTE}" -eq 1 ]; then
        sched_per_node_prefix || exit 1
    fi
    launch_cmd=("${PREFIX[@]}" python "${TRAIN_SCRIPT}" "${TRAIN_ARGS[@]}")

elif [ "${LAUNCHER}" = "torchrun" ]; then
    if [ "${NNODES}" -eq 1 ] && [ "${REMOTE}" -eq 0 ]; then
        launch_cmd=(
            torchrun --standalone --nproc_per_node="${GPUS_PER_NODE}"
            "${TRAIN_SCRIPT}" "${TRAIN_ARGS[@]}"
        )
    else
        # One torchrun per node; c10d rendezvous assigns node ranks
        sched_per_node_prefix || exit 1
        launch_cmd=(
            "${PREFIX[@]}"
            torchrun
            --nnodes="${NNODES}"
            --nproc_per_node="${GPUS_PER_NODE}"
            --rdzv_id="${JOB_NUM:-particleman}"
            --rdzv_backend=c10d
            --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}"
            "${TRAIN_SCRIPT}" "${TRAIN_ARGS[@]}"
        )
    fi

elif [ "${LAUNCHER}" = "mpiexec" ] || [ "${LAUNCHER}" = "srun" ]; then
    # One rank per GPU; ranks are read from launcher env vars in setup_distributed
    sched_per_gpu_prefix "${LAUNCHER}" || exit 1
    launch_cmd=("${PREFIX[@]}" python "${TRAIN_SCRIPT}" "${TRAIN_ARGS[@]}")

else
    echo "ERROR: unknown LAUNCHER=${LAUNCHER} (use torchrun, mpiexec or srun)" >&2
    exit 1
fi

# ============================================================================
# Run (batch mode also records the command and job output in the run dir)
# ============================================================================

RUN_DIR=""
if [ "${BATCH}" -eq 1 ] && [ -n "${JOB_NUM}" ] && [ -n "${EXPERIMENT_NAME}" ]; then
    RUN_DIR="${OUTPUT_DIR:-outputs}/${EXPERIMENT_NAME}/${RUN_NAME:-${TIMESTAMP}_${SCHED_NAME}${JOB_NUM}}"
    mkdir -p "${RUN_DIR}"
    printf '%q ' "${launch_cmd[@]}" > "${RUN_DIR}/launch_command.sh"
    echo >> "${RUN_DIR}/launch_command.sh"
    echo "Run directory: ${RUN_DIR}"
fi

echo "Launch command: ${launch_cmd[*]}"
"${launch_cmd[@]}"
status=$?

echo ""
echo "=============================================="
echo "Finished at: $(date) (exit ${status})"
echo "=============================================="

if [ -n "${RUN_DIR}" ]; then
    # <job_name>.o<job_id>: PBS -j oe default, Slurm --output=%x.o%j
    JOB_OUTPUT_FILE="${WORKDIR}/${JOB_NAME}.o${JOB_NUM}"
    if [ -f "${JOB_OUTPUT_FILE}" ]; then
        cp "${JOB_OUTPUT_FILE}" "${RUN_DIR}/job_output.log"
    fi
fi

exit ${status}
