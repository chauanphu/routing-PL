import time
from meta.solver import Experiment, Node, Problem, print_routes
from meta.GreyWolf import GreyWolfOptimization as GWO
from meta.PSO import ParticleSwarmOptimization as PSO
# Create a problem instance
instance = Problem()
instance.load_data("data/25/C101_co_25.txt")

start_time = time.time()
# Load the solver
gwo = GWO(instance.evaluate_position, num_wolves=100, num_iterations=200, search_space=instance.get_search_space(), local_search_iterations=None)
gwo_ls = GWO(instance.evaluate_position, num_wolves=50, num_iterations=200, search_space=instance.get_search_space(), local_search_iterations=10)
# pso = PSO(instance.evaluate_position, num_particles=100, num_iterations=200, search_space=instance.get_search_space())
solvers = [(gwo, "GWO"), (gwo_ls, "GWO_LS")]
experiments = Experiment(instance=instance, solvers=solvers)
experiments.run()
experiments.report()
# gwo.optimize()
# routes: list[list[Node]]
# distance, routes = instance.position2route(gwo.global_best_position)
# print("Distance: ", distance)
# print_routes(routes)