#!/usr/bin/env bash
# Dispatches jobs from jobs.txt across N GPUs (default: 2).
# Usage: ./run_jobs.sh [NUM_GPUS [JOBS_PER_GPU [JOBS_FILE]]]
#        ./run_jobs.sh jobs.txt
#
# Each GPU runs JOBS_PER_GPU jobs concurrently (default: 1).

if [[ -f "$1" ]]; then
    NUM_GPUS=2; JOBS_PER_GPU=1; JOBS_FILE=$1
elif [[ "$1" =~ ^[0-9]+$ ]]; then
    NUM_GPUS=$1
    if [[ "$2" =~ ^[0-9]+$ ]]; then
        JOBS_PER_GPU=$2; JOBS_FILE=${3:-jobs.txt}
    else
        JOBS_PER_GPU=1; JOBS_FILE=${2:-jobs.txt}
    fi
else
    NUM_GPUS=2; JOBS_PER_GPU=1; JOBS_FILE=jobs.txt
fi

NUM_SLOTS=$((NUM_GPUS * JOBS_PER_GPU))
LOG_DIR="logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

mapfile -t JOBS < <(grep -v '^\s*#' "$JOBS_FILE" | grep -v '^\s*$')
TOTAL=${#JOBS[@]}

if [[ $TOTAL -eq 0 ]]; then
    echo "No jobs found in $JOBS_FILE"
    exit 1
fi

echo "Dispatching $TOTAL jobs across $NUM_GPUS GPUs ($JOBS_PER_GPU job(s)/GPU) — logs in $LOG_DIR"

job_idx=0
declare -a SLOT_PIDS  # PID running on each slot (-1 = free)
declare -a SLOT_JOB   # job index assigned to each slot

for ((s=0; s<NUM_SLOTS; s++)); do SLOT_PIDS[$s]=-1; done

run_job() {
    local slot=$1 idx=$2 cmd=$3
    local gpu=$((slot % NUM_GPUS))
    local logfile="$LOG_DIR/gpu${gpu}_slot${slot}_job$(printf '%03d' "$idx").log"
    echo "[GPU $gpu / slot $slot] Starting job $((idx+1))/$TOTAL: $cmd"
    CUDA_VISIBLE_DEVICES=$gpu bash -c "$cmd" > "$logfile" 2>&1 &
    SLOT_PIDS[$slot]=$!
    SLOT_JOB[$slot]=$idx
}

# Seed: fill all slots
for ((s=0; s<NUM_SLOTS && job_idx<TOTAL; s++, job_idx++)); do
    run_job "$s" "$job_idx" "${JOBS[$job_idx]}"
done

# Loop until all jobs done
while true; do
    all_done=true
    for ((s=0; s<NUM_SLOTS; s++)); do
        pid=${SLOT_PIDS[$s]}
        [[ $pid -eq -1 ]] && continue
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid"; status=$?
            idx=${SLOT_JOB[$s]}
            echo "[slot $s] Job $((idx+1))/$TOTAL finished (exit $status)"
            SLOT_PIDS[$s]=-1
            if [[ $job_idx -lt $TOTAL ]]; then
                run_job "$s" "$job_idx" "${JOBS[$job_idx]}"
                ((job_idx++))
            fi
        fi
        [[ ${SLOT_PIDS[$s]} -ne -1 ]] && all_done=false
    done
    $all_done && break
    sleep 5
done

echo "All $TOTAL jobs completed. Logs: $LOG_DIR"