from meta.solver import Problem
from experiment import Experiment
from meta.ACO import AntColonyOptimization as ClassicACO, SACO, PACO
from meta.SA import SimulatedAnnealing as SA
from meta.GreyWolf import GreyWolfOptimization as GWO

# Create a problem instance
instance = Problem()
instance.load_data("data/100/C101_co_100.txt")

# Define your solvers and parameters:
solvers_to_test = {
    "Sequential ACO": (
        SACO,  
        {"num_ants": 1000, "alpha": 1.0, "beta": 1.0, "evaporation_rate": 0.1, "Q": 1.0, "num_iterations": 100}
    )
}

# Create an Experiment instance.
experiment = Experiment(problem=instance,
                        solvers_dict=solvers_to_test,
                        num_runs=10)

# Run experiments sequentially.
results = experiment.run()

# Write a CSV report for each solver.
experiment.write_csv_report()

# Optionally, print aggregated results.
for solver in solvers_to_test.keys():
    summary = experiment.aggregate_results(solver)
    if summary is not None:
        print(f"Aggregated results for {solver}: {summary}")
    else:
        print(f"No results available for {solver}.")