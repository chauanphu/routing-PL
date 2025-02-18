from meta.solver import Problem, Experiment
from meta.ACO import AntColonyOptimization as ClassicACO, SACO, PACO
from meta.SA import SimulatedAnnealing as SA
from meta.GreyWolf import GreyWolfOptimization as GWO
# Create a problem instance
instance = Problem()
instance.load_data("data/50/C101_co_50.txt")

experiment = Experiment(
    instance, 
    solvers=[
        (SACO(instance, num_ants=1000, num_iterations=100, alpha=1.0, beta=1.0, evaporation_rate=0.1, Q=1.0), "SACO"),
        (PACO(instance, num_ants=1000, batch_size=1000, num_iterations=100, alpha=1.0, beta=1.0, evaporation_rate=0.1, Q=1.0), "PACO"),
    ],
    num_experiments=4
    )

# Run the experiment
experiment.run()

# Export
experiment.report()