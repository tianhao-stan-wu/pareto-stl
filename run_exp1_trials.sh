#!/bin/bash

set -u

NUM_RUNS=2
START_SEED=0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pareto-stl

echo "Starting CARLA..."

./CARLA_0.9.15/CarlaUE4.sh \
    > carla_server.log 2>&1 &

CARLA_PID=$!

echo "CARLA PID: $CARLA_PID"

# Give CARLA time to initialize.
sleep 10

# --------------------------------------------------
# Cleanup
# --------------------------------------------------

cleanup() {
    echo "Stopping CARLA..."
    kill "$CARLA_PID" 2>/dev/null || true
}

# Stop CARLA when this script exits or is interrupted.
trap cleanup EXIT INT TERM

# --------------------------------------------------
# Run experiments
# --------------------------------------------------

mkdir -p logs/exp1_batch

for ((i=0; i<NUM_RUNS; i++)); do
    seed=$((START_SEED + i))

    echo "========================================"
    echo "Trial $((i + 1)) / $NUM_RUNS"
    echo "Seed: $seed"
    echo "========================================"

    python run_exp1.py "$seed" 2>&1 \
        | tee "logs/exp1_batch/trial_${i+1}.log"

    status=${PIPESTATUS[0]}

    if [ "$status" -eq 0 ]; then
        echo "Trial $((i + 1)) completed."
    else
        echo "Trial $((i + 1)) FAILED with exit code $status."
    fi
done

echo "========================================"
echo "All $NUM_RUNS trials finished."
echo "========================================"