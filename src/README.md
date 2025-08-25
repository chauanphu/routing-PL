# Sample commands

**Run the sensitivity analysis**
`uv run -m src.experiment.sensitivity`:
 - `--size <large | medium | small>`: Size of the problem instance.
 - `--instance-file <path_to_instance_file>`: Path to the instance file to use for the experiment.
 - `--parameters <path_to_parameters_file>`: Path to the YAML file containing the parameters for the experiment.
 - `--output <path_to_output_file>`: Path to the output CSV file to store the results.
 - `--num-runs <number_of_runs>`: Number of runs to perform for each parameter setting.

For example: `uv run -m src.experiment.sensitivity --size small --instance-file data/25 --parameters parameters/paco.tune.yaml --output results/sensitivity.csv --num-runs 5`

**Run the solver on each instance**

1. `cd src/build`
2. `make`
3. `cd ..`
4. `./build/main`
 - `--solver <paco | sa>`: The solver PACO (proposed model), SA - Simulated Annealing
 - `--params <path_to_parameters_file>`: Path to the YAML file containing the parameters for the solver.
 - `--instances <path_to_instance_files.txt>`: Path to the directory containing the instance files.
 - `--num-runs <number_of_runs>`: Number of runs to perform for each instance.
 - `--output <path_to_output_file.csv>`: Path to the output CSV file to store the results.
 - `--size <large | medium | small>`: Size of the problem instance.
 - `--verbose`: Enable verbose output. (Default is result only)

 For example: `./build/main --solver paco --params parameters/paco.tune.yaml --instances data/25/C101_co_25.txt --num-runs 1 --output output/test.csv --size small --verbose 1`