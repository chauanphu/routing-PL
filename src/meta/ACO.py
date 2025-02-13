import math
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor

if __name__ == "__main__":
    from solver import Problem, Solver, Node
else:
    from meta.solver import Problem, Solver, Node

def euclidean_distance(p1, p2):
    """Compute Euclidean distance between two 2D points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

class AntColonyOptimization(Solver):
    def __init__(self, problem: Problem, num_ants=1000, num_iterations=100, batch_size=100, alpha=1.0, beta=2.0, evaporation_rate=0.1, Q=1.0):
        """
        Initializes the ACO optimizer for the VRPPL problem using a 3D pheromone matrix and sets up key parameters for
        ant colony optimization with batch processing for parallel execution.

        Parameters:
            problem (Problem): The routing problem instance containing customer data and related properties.
            num_ants (int, optional): The number of ants to simulate per iteration, influencing exploration. Defaults to 1000.
            num_iterations (int, optional): The number of iterations the algorithm will perform to refine solutions. Defaults to 100.
            batch_size (int, optional): The number of iterations to process in each parallel batch. Defaults to 100.
            alpha (float, optional): The factor controlling the influence of pheromone trails in ant decision-making. Defaults to 1.0.
            beta (float, optional): The factor controlling the influence of heuristic information (e.g., distance) in ant decisions. Defaults to 2.0.
            evaporation_rate (float, optional): The rate at which pheromone intensity diminishes over time, promoting exploration. Defaults to 0.1.
        """
        super().__init__(problem.node2routes, num_iterations=num_iterations)
        self.problem: Problem = problem
        self.num_iterations = num_iterations
        self.num_ants = num_ants
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.Q = Q

        # n: number of customers
        self.n = self.problem.num_customers

        # Initialize a single 3D pheromone matrix with dimensions (n+1) x (n+1) x 2.
        self.pheromones = np.ones((self.n + 1, self.n + 1, 2))
        for j, customer in enumerate(self.problem.customers, start=1):
            if customer.customer_type == 1:
                self.pheromones[:, j, 1] = 0.0
            elif customer.customer_type == 2:
                self.pheromones[:, j, 0] = 0.0

        self.global_best_fitness = float('inf')
        self.global_best_solution = None
        self.global_best_routes = None
        self.fitness_history = []

    def construct_solution(self):
        """
        Constructs one ant's solution using the integrated 3D pheromone matrix.
        """
        current_location = (self.problem.depot.x, self.problem.depot.y)
        prev_index = 0
        unvisited = list(range(1, self.n + 1))
        transitions = []
        final_route = [self.problem.depot]

        while unvisited:
            candidate_options = []
            for j in unvisited:
                customer = self.problem.customers[j - 1]
                allowed = [0] if customer.customer_type == 1 else [1] if customer.customer_type == 2 else [0, 1]
                for d in allowed:
                    candidate_coord = (customer.x, customer.y) if d == 0 else (customer.assigned_locker.x, customer.assigned_locker.y)
                    distance = euclidean_distance(current_location, candidate_coord)
                    heuristic = 1.0 / distance if distance > 0 else 1e6
                    tau = self.pheromones[prev_index, j, d]
                    value = (tau ** self.alpha) * (heuristic ** self.beta)
                    candidate_options.append((j, d, value, candidate_coord))

            total_value = sum(option[2] for option in candidate_options)
            probs = [option[2] / total_value for option in candidate_options] if total_value > 0 else [1 / len(candidate_options)] * len(candidate_options)
            r = random.random()
            cumulative = 0.0
            selected = None
            for (j, d, _, candidate_coord), prob in zip(candidate_options, probs):
                cumulative += prob
                if r <= cumulative:
                    selected = (j, d, candidate_coord)
                    break
            if selected is None:
                selected = candidate_options[-1][0:2] + (candidate_options[-1][3],)

            j, d, candidate_coord = selected
            transitions.append((prev_index, j, d))
            final_route.append(self.problem.customers[j - 1] if d == 0 else self.problem.customers[j - 1].assigned_locker)
            current_location = candidate_coord
            prev_index = j
            unvisited.remove(j)
        final_route.append(self.problem.depot)
        return transitions, final_route

    def update_pheromones(self, solutions):
        """Updates the integrated 3D pheromone matrix based on the ant solutions."""
        self.pheromones *= (1 - self.evaporation_rate)
        for transitions, fitness in solutions:
            deposit = self.Q / (fitness if fitness > 0 else 1e-8)
            for (i, j, d) in transitions:
                self.pheromones[i, j, d] += deposit

        for j, customer in enumerate(self.problem.customers, start=1):
            if customer.customer_type == 1:
                self.pheromones[:, j, 1] = 0.0
            elif customer.customer_type == 2:
                self.pheromones[:, j, 0] = 0.0

    def single_ant_solution(self):
        """Constructs and evaluates a single ant's solution."""
        transitions, final_route = self.construct_solution()
        fitness, routes = self.problem.node2routes(final_route)
        return transitions, fitness, final_route, routes

    def optimize(self, verbose=True):
        """Executes the ACO optimization process using parallel processing."""
        with ProcessPoolExecutor() as executor:
            for iteration in range(self.num_iterations):
                solutions = []
                for i in range(0, self.num_ants, self.batch_size):
                    futures = [executor.submit(self.single_ant_solution) for _ in range(self.batch_size)]
                    batch_results = [future.result() for future in futures]
                    solutions.extend(batch_results)
                self.update_pheromones([(transitions, fitness) for (transitions, fitness, _, _) in solutions])
                for (transitions, fitness, final_route, routes) in solutions:
                    if fitness < self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best_solution = final_route
                        self.global_best_routes = routes
                self.fitness_history.append(self.global_best_fitness)
                if verbose:
                    print(f"Iteration {iteration + 1}: Best Fitness = {self.global_best_fitness}")
        return self.global_best_solution, self.global_best_fitness, self.global_best_routes


if __name__ == '__main__':
    import time
    from solver import Node, Problem, print_routes
    # Create a problem instance
    instance = Problem()
    instance.load_data("data/25/R102_co_25.txt")

    start_time = time.time()
    # Load the solver
    aco = AntColonyOptimization(problem=instance, num_ants=200, num_iterations=50, batch_size=60, alpha=1.0, beta=1.0, evaporation_rate=0.1, Q=1.0)
    aco.optimize()
    print("Elapsed time (s):", time.time() - start_time)
    print("Distance: ", aco.global_best_fitness)
    print([node.node_id for node in aco.global_best_solution])
    print_routes(aco.global_best_routes)
    aco.plot_fitness_history()
    aco.plot_routes()