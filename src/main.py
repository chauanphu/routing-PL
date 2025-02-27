from meta.solver import Problem
from experiment import Experiment
from meta.ACO import AntColonyOptimization as ClassicACO, SACO, PACO
from meta.SA import SimulatedAnnealing as SA
from meta.GreyWolf import GreyWolfOptimization as GWO

def log_to_file(log, filename):
    print(log)
    with open(filename, "a") as f:
        f.write(log)
    
# Define your solvers and parameters:
solvers_to_test = {
    # "Sequential_ACO": (
    #     SACO,  
    #     {"num_ants": 1000, "alpha": 1.0, "beta": 1.0, "evaporation_rate": 0.1, "Q": 1.0, "num_iterations": 100}
    # ),
    "Parallel ACO": (
        PACO,  
        {"num_ants": 3000, "batch_size": 100, "alpha": 1.0, "beta": 1.0, "evaporation_rate": 0.2, "Q": 1.0, "num_iterations": 100, "elitist_num": 5}
    ),
}

# List all instances file in data/100 folder
import os
instances = os.listdir("data/100")
instances.sort()
# Exclude 1 to 6
# except_instances = []
# instances = [instance for instance in instances if instance not in except_instances]
instances = instances[9:34]
for instance in instances:
    print(instance)
# Sort the instances, by name ascending
# Solve all instances in data/100 folder
for instance_file in instances:
    try:
        instance_name = instance_file.split(".")[0]
        instance = Problem()
        instance.load_data(f"data/100/{instance_file}")
        experiment = Experiment(problem=instance, solvers_dict=solvers_to_test, num_runs=10)
        results = experiment.run()
        experiment.write_csv_report(problem_name=instance_name)
    except Exception as e:
        print(f"Error: {e}\n")
        continue