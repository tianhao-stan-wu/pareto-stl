#!/bin/bash

set -u

DENSITY="${1:-10}"
NUM_RUNS="${2:-10}"
START_SEED=0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pareto-stl

BATCH_DIR="logs/exp2_batch_d${DENSITY}_${NUM_RUNS}"
mkdir -p "$BATCH_DIR"
mkdir -p "$BATCH_DIR/log"
mkdir -p "$BATCH_DIR/trials"

echo "Density: $DENSITY"
echo "Number of trials: $NUM_RUNS"
echo "Starting CARLA..."

./CARLA_0.9.15/CarlaUE4.sh \
    > "$BATCH_DIR/carla_server.log" 2>&1 &

CARLA_PID=$!

echo "CARLA PID: $CARLA_PID"
echo "Waiting for CARLA to initialize..."

sleep 10

cleanup() {
    echo "Stopping CARLA..."
    kill "$CARLA_PID" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

for ((i=0; i<NUM_RUNS; i++)); do

    seed=$((START_SEED + i))

    LOG_FILE="$BATCH_DIR/log/trial_${i}.log"
    DONE_FILE="$BATCH_DIR/log/trial_${i}.done"

    # Skip trials that already completed successfully
    if [ -f "$DONE_FILE" ]; then
        echo "Trial $((i + 1)) / $NUM_RUNS already completed. Skipping."
        continue
    fi

    echo "========================================"
    echo "Trial $((i + 1)) / $NUM_RUNS"
    echo "Density: $DENSITY"
    echo "Seed: $seed"
    echo "========================================"

    python run_exp2.py "$seed" "$DENSITY" "$NUM_RUNS" 2>&1 \
        | tee "$LOG_FILE"

    status=${PIPESTATUS[0]}

    if [ "$status" -eq 0 ]; then
        touch "$DONE_FILE"
        echo "Trial $((i + 1)) completed successfully."
    else
        echo "Trial $((i + 1)) FAILED with exit code $status."
    fi

done

echo "========================================"
echo "All $NUM_RUNS Exp. 2 trials finished."
echo "Density: $DENSITY"
echo "========================================"