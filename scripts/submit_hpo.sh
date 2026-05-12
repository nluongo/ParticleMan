#!/bin/bash
# Submit N parallel HPO workers. All workers share the same Optuna study via
# a SQLite file on the shared filesystem, so Optuna coordinates trial
# assignment automatically.
#
# Usage:
#   bash scripts/submit_hpo.sh [N_WORKERS]
#
# Example — 4 workers, each running 5 trials (20 trials total in parallel):
#   bash scripts/submit_hpo.sh 4
#
# Note: STORAGE must be on a shared POSIX filesystem (GPFS/Lustre) that
# supports flock. If file locking is unavailable, replace sqlite:/// with a
# PostgreSQL or MySQL URL.

N_WORKERS=${1:-4}

# ── Edit these to match your run ──────────────────────────────────────────────
STUDY="particleman_hpo"
STORAGE="sqlite:////lcrc/group/ATLAS/users/nluongo/ParticleMan/hpo/${STUDY}.db"
DATA_CONFIG="bbllv08_classify"
MODE="classify"
EPOCHS=5
TRIALS_PER_WORKER=5   # total trials = N_WORKERS * TRIALS_PER_WORKER
MAX_EVENTS=50000
# ─────────────────────────────────────────────────────────────────────────────

HPO_ARGS="--study-name ${STUDY} --storage ${STORAGE} --data-config ${DATA_CONFIG} --mode ${MODE} --epochs ${EPOCHS} --trials ${TRIALS_PER_WORKER} --max-events ${MAX_EVENTS}"

echo "Submitting ${N_WORKERS} HPO workers"
echo "  Study   : ${STUDY}"
echo "  Storage : ${STORAGE}"
echo "  Config  : ${DATA_CONFIG}, mode=${MODE}"
echo "  Per worker: ${TRIALS_PER_WORKER} trials x ${EPOCHS} epochs"
echo "  Total trials: $((N_WORKERS * TRIALS_PER_WORKER))"
echo ""

# Ensure the directory for the SQLite DB exists
mkdir -p "$(dirname "${STORAGE#sqlite:////}")" 2>/dev/null || true

for i in $(seq 1 "${N_WORKERS}"); do
    JOB_ID=$(qsub -l select=1:ngpus=1 \
                  -v HPO_ARGS="${HPO_ARGS}" \
                  scripts/pbs/submit_hpo.pbs)
    echo "  Worker ${i}: ${JOB_ID}"
done

echo ""
echo "All workers submitted. Monitor with: qstat -u \$USER"
