SOLVER="aco"
PARAM_FILE="parameters/${SOLVER}.param.yaml"
TEST_EXEC="build/main"
N_RUNS=5 # A reasonable number for BO

echo "--- Solving small instances (data/25) ---"
$TEST_EXEC --solver $SOLVER --params $PARAM_FILE --instances data/25 --num-runs $N_RUNS --size small --output output/solutions/small_$SOLVER.csv

echo "--- Solving medium instances (data/50) ---"
$TEST_EXEC --solver $SOLVER --params $PARAM_FILE --instances data/50 --num-runs $N_RUNS --size medium --output output/solutions/medium_$SOLVER.csv

echo "--- Solving large instances (data/100) ---"
$TEST_EXEC --solver $SOLVER --params $PARAM_FILE --instances data/100 --num-runs $N_RUNS --size large --output output/solutions/large_$SOLVER.csv