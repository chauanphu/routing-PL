SOLVER="alns"
PARAM_FILE="parameters/${SOLVER}.param.yaml"
TEST_EXEC="build/main"
N_RUNS=4 # A reasonable number for BO

# echo "--- Solving small instances (data/25) ---"
# $TEST_EXEC --solver $SOLVER --params $PARAM_FILE --instances data/25 --num-runs $N_RUNS --size small --output output/solutions/small_alns.csv

# echo "--- Solving medium instances (data/50) ---"
# $TEST_EXEC --solver $SOLVER --params $PARAM_FILE --instances data/50 --num-runs $N_RUNS --size medium --output output/solutions/medium_alns.csv

echo "--- Solving large instances (data/100) ---"
$TEST_EXEC --solver $SOLVER --params $PARAM_FILE --instances data/100 --num-runs $N_RUNS --size large --output output/solutions/large_alns.csv