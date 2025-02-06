import time
from meta.solver import Node, Problem, print_routes
from meta.GreyWolf import GreyWolfOptimization as GWO
from meta.PSO import ParticleSwarmOptimization as PSO
# Create a problem instance
instance = Problem()
instance.load_data("data/25/C101_co_25.txt")

start_time = time.time()
# Load the solver
gwo = GWO(problem=instance, num_wolves=1000, num_iterations=100, local_search_iterations=None)

gwo.optimize()
routes: list[list[Node]]
print("Distance: ", gwo.global_best_fitness)
print_routes(gwo.global_best_routes)
gwo.plot_fitness_history()