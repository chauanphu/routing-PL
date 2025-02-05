import time
from meta.solver import Experiment, Node, Problem, print_routes
from meta.GreyWolf import GreyWolfOptimization as GWO
from meta.PSO import ParticleSwarmOptimization as PSO
# Create a problem instance
instance = Problem()
instance.load_data("data/25/C101_co_25.txt")

start_time = time.time()
# Load the solver
gwo = GWO(instance.evaluate_position, num_wolves=500, num_iterations=100, search_space=instance.get_search_space())
# pso = PSO(instance.evaluate, num_particles=10, num_iterations=100, search_space=instance.get_search_space())
# solvers = [gwo, pso]
# experiments = Experiment(instance=instance, solvers=solvers)
# experiments.run()
# experiments.report()
gwo.optimize()
routes: list[list[Node]]
distance, routes = instance.position2route(gwo.global_best_position)
print("Distance: ", distance)
print_routes(routes)