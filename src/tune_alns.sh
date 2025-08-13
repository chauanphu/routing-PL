#!/bin/bash

# Script to run Bayesian optimization for ALNS solver for different data sizes.

SOLVER="alns"
TUNE_FILE="parameters/alns.tune.yaml"
PARAM_FILE="parameters/alns.param.yaml"
TEST_EXEC="build/test"
N_CALLS=50 # A reasonable number for BO

# # Small
# echo "--- Tuning for small instances (data/25) ---"
# uv run src/experiment/bao_tune.py \
#     --solver $SOLVER \
#     --tune-file $TUNE_FILE \
#     --param-file $PARAM_FILE \
#     --runtime-weight 1.0 \
#     --instance-dir data/25 \
#     --size small \
#     --n-calls $N_CALLS \
#     --output output/tuning/alns_small_best_params.json \
#     --test-exec $TEST_EXEC

# # Medium
# echo "--- Tuning for medium instances (data/50) ---"
# uv run src/experiment/bao_tune.py \
#     --solver $SOLVER \
#     --tune-file $TUNE_FILE \
#     --param-file $PARAM_FILE \
#     --runtime-weight 0.5 \
#     --instance-dir data/50 \
#     --size medium \
#     --n-calls $N_CALLS \
#     --output output/tuning/alns_medium_best_params.json \
#     --test-exec $TEST_EXEC

# Large
echo "--- Tuning for large instances (data/100) ---"
uv run src/experiment/bao_tune.py \
    --solver $SOLVER \
    --tune-file $TUNE_FILE \
    --param-file $PARAM_FILE \
    --runtime-weight 1.0 \
    --instance-dir data/100 \
    --size large \
    --n-calls $N_CALLS \
    --output output/tuning/alns_large_best_params.json \
    --test-exec $TEST_EXEC

echo "--- ALNS tuning complete ---"
