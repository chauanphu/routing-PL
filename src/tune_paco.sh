#!/bin/bash

# Script to run Bayesian optimization for PACO solver on small instances.
# Run this script from the workspace root.

SOLVER="paco"
TUNE_FILE="src/parameters/${SOLVER}.tune.yaml"
PARAM_FILE="src/parameters/${SOLVER}.param.yaml"
OUTPUT_FILE="src/output/tuning/${SOLVER}.small.json"
TEST_EXEC="src/build/test"
INSTANCE_DIR="dataset/small" # User requested dataset/small
SIZE="small"
N_CALLS=30

# Ensure output directory exists
mkdir -p $(dirname "$OUTPUT_FILE")

echo "--- Tuning $SOLVER for $SIZE instances in $INSTANCE_DIR ---"

uv run src/src/experiment/bao_tune.py \
    --solver "$SOLVER" \
    --tune-file "$TUNE_FILE" \
    --param-file "$PARAM_FILE" \
    --instance-dir "$INSTANCE_DIR" \
    --size "$SIZE" \
    --n-calls "$N_CALLS" \
    --output "$OUTPUT_FILE" \
    --test-exec "$TEST_EXEC" \
    --n-samples 5 \
    --timeout 100 \
    --runtime-weight 1.0
