from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import shared_memory

if __name__ == "__main__":
    from solver import Problem, Solver, Node, print_routes
else:
    from meta.solver import Problem, Solver, Node, print_routes

def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def export_pheromones_heatmap(pheromones, filename="pheromones_heatmap.png"):
    """
    Plots both layers of the 3D pheromone matrix as heatmaps.
    
    Args:
        pheromones: 3D numpy array of shape (n+1, n+1, 2)
        title_prefix: String to prepend to plot titles
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot direct delivery layer (mode 0)
    im1 = ax1.imshow(pheromones[:, :, 0], cmap='hot', interpolation='nearest')
    ax1.set_title(f"Home Delivery Pheromones")
    plt.colorbar(im1, ax=ax1)
    
    # Plot locker delivery layer (mode 1)
    im2 = ax2.imshow(pheromones[:, :, 1], cmap='hot', interpolation='nearest')
    ax2.set_title(f"Locker Delivery Pheromones")
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

class AntColonyOptimization(Solver):
    def __init__(self, problem: Problem, num_ants=20, num_iterations=100,
                 alpha=1.0, beta=2.0, evaporation_rate=0.1, Q=1.0, p=0.5):
        """
        Initializes the ACO optimizer with pre-assigned delivery nodes.
        For each customer, we apply the locker assignment (with explore=False) so that
        if a customer is type II or III with locker preference, the ant will visit the
        assigned locker rather than the customer’s house.
        
        Parameters:
            problem (Problem): The VRP problem instance.
            num_ants (int): Number of ants per iteration.
            num_iterations (int): Total number of iterations.
            alpha (float): Relative importance of pheromone.
            beta (float): Relative importance of heuristic (1/distance).
            evaporation_rate (float): Rate at which pheromone evaporates.
            Q (float): Constant for pheromone deposit.
        """
        super().__init__(problem.node2routes, num_iterations=num_iterations)
        self.problem: Problem = problem
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.Q = Q
        self.p = p

        # Determine the number of customers.
        self.n = self.problem.num_customers

        # -------------------------------
        # Pre-assign delivery nodes:
        # -------------------------------
        # For each customer, apply locker assignment before constructing the route.
        # For type II/III with locker preference, the assigned locker will be used;
        # for type I, the customer node will be used.
        self.delivery_nodes = []
        for customer in self.problem.customers:
            chosen_node, _ = self.problem._locker_assignment(customer, explore=False)
            self.delivery_nodes.append(chosen_node)

        # Create the list of nodes used for ACO:
        # Index 0 corresponds to the depot;
        # indices 1...n correspond to the pre-assigned delivery nodes.
        self.nodes = [self.problem.depot] + self.delivery_nodes

        # Initialize pheromone matrix with a small positive constant.
        self.pheromones = np.ones((self.n + 1, self.n + 1))

        # Compute heuristic information: eta[i][j] = 1 / distance(i,j) for i != j.
        self.eta = np.zeros((self.n + 1, self.n + 1))
        for i in range(self.n + 1):
            for j in range(self.n + 1):
                if i == j:
                    self.eta[i][j] = 0.0
                else:
                    d = self.problem.euclidean_distance(self.nodes[i], self.nodes[j])
                    self.eta[i][j] = 1.0 / d if d != 0 else 1e6

        self.global_best_fitness = float('inf')
        self.best_solution = None  # Will be a list of Node objects.
        self.global_best_routes = None
        self.fitness_history = []

    def construct_solution(self) -> list:
        """
        Constructs a solution for one ant.
        The ant builds a route (a permutation) over the pre-assigned delivery nodes.
        The returned solution is a list of Node objects, with the depot at the beginning
        and at the end.
        
        Returns:
            solution (list[Node]): The constructed route.
        """
        current = 0  # Start at the depot (index 0)
        unvisited = list(range(1, self.n + 1))  # Indices for delivery nodes (1...n)
        solution_indices = [current]

        while unvisited:
            probabilities = []
            # Calculate the probability of moving from the current node to each candidate.
            for j in unvisited:
                tau = self.pheromones[current][j] ** self.alpha
                eta = self.eta[current][j] ** self.beta
                probabilities.append(tau * eta)
            total = sum(probabilities)
            if total == 0:
                probabilities = [1 / len(unvisited)] * len(unvisited)
            else:
                probabilities = [p / total for p in probabilities]

            # Roulette wheel selection.
            r = random.random()
            cumulative = 0.0
            for idx, j in enumerate(unvisited):
                cumulative += probabilities[idx]
                if r <= cumulative:
                    next_index = j
                    break

            solution_indices.append(next_index)
            unvisited.remove(next_index)
            current = next_index

        # Complete the tour by returning to the depot.
        solution_indices.append(0)

        # Map indices to Node objects.
        # Note: index 0 is the depot; indices 1..n come from self.delivery_nodes.
        solution = []
        for idx in solution_indices:
            if idx == 0:
                solution.append(self.problem.depot)
            else:
                solution.append(self.delivery_nodes[idx - 1])
        return solution

    def get_delivery_index(self, node: Node) -> int:
        """
        Given a node (delivery node), return its index in the pre-assigned delivery_nodes list.
        The depot is always index 0.
        """
        for i, d in enumerate(self.delivery_nodes):
            if d.node_id == node.node_id:
                return i + 1
        return 0

    def update_pheromones(self, all_solutions: list[tuple[list, float]]):
        """
        Updates the pheromone matrix based on the solutions generated by all ants.
        Each solution deposits pheromone inversely proportional to its fitness.
        
        Parameters:
            all_solutions (list): A list of tuples (solution, fitness) where each solution
                                  is a list of Node objects.
        """
        # Evaporate pheromone.
        self.pheromones *= (1 - self.evaporation_rate)

        # Deposit pheromone for each solution.
        for solution, fitness in all_solutions:
            deposit = self.Q / (fitness if fitness > 0 else 1e-8)
            indices = []
            # Map each node in the solution back to its index.
            for node in solution:
                if node.node_id == self.problem.depot.node_id:
                    indices.append(0)
                else:
                    indices.append(self.get_delivery_index(node))
            for i in range(len(indices) - 1):
                a = indices[i]
                b = indices[i + 1]
                self.pheromones[a][b] += deposit
                self.pheromones[b][a] += deposit

        # Optional: Elitist update for the best solution found so far.
        if self.best_solution is not None:
            deposit = self.Q / (self.global_best_fitness if self.global_best_fitness > 0 else 1e-8)
            indices = []
            for node in self.best_solution:
                if node.node_id == self.problem.depot.node_id:
                    indices.append(0)
                else:
                    indices.append(self.get_delivery_index(node))
            for i in range(len(indices) - 1):
                a = indices[i]
                b = indices[i + 1]
                self.pheromones[a][b] += deposit
                self.pheromones[b][a] += deposit

    def optimize(self, verbose=True):
        """
        Executes the ACO optimization process.
        
        Returns:
            best_permutation (list[Node]): Best route found.
            best_fitness (float): Fitness value of the best route.
            best_routes: The routes corresponding to the best route.
        """
        for iteration in range(self.num_iterations):
            ## 1. Apply seeding solution
            initial_solution = self.problem.initialize_solution(p=self.p)[1]
            initial_fitness, iniital_routes = self.problem.node2routes(initial_solution)
            all_solutions = [(initial_solution, initial_fitness)]
            if initial_fitness < self.global_best_fitness:
                    self.global_best_fitness = initial_fitness
                    self.best_solution = initial_solution
                    self.global_best_routes = iniital_routes
            ## 2. Construct solutions for each ant.
            for ant in range(self.num_ants):
                solution: list[Node] = self.construct_solution()
                # Evaluate the solution using the problem's evaluation method.
                fitness, routes = self.problem.node2routes(solution)
                all_solutions.append((solution, fitness))
                # Update global best if necessary.
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.best_solution = solution
                    self.global_best_routes = routes
            self.update_pheromones(all_solutions)
            self.fitness_history.append(self.global_best_fitness)
            if verbose:
                print(f"Iteration {iteration + 1}: Best Fitness = {self.global_best_fitness}")
        return self.best_solution, self.global_best_fitness, self.global_best_routes
    
class SACO(Solver):
    def __init__(self, problem: Problem, num_ants=20, num_iterations=100,
                 alpha=1.0, beta=2.0, evaporation_rate=0.1, Q=1.0):
        """
        Initializes the Sequential ACO optimizer with 3D pheromone.
        """
        super().__init__(problem.node2routes, num_iterations=num_iterations)
        self.problem: Problem = problem
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.Q = Q

        self.n = self.problem.num_customers

        # Initialize 3D pheromone matrix.
        self.pheromones = np.ones((self.n + 1, self.n + 1, 2))
        for j, customer in enumerate(self.problem.customers, start=1):
            if customer.customer_type == 1:
                self.pheromones[:, j, 1] = 0.0
            elif customer.customer_type == 2:
                self.pheromones[:, j, 0] = 0.0

        self.global_best_fitness = float('inf')
        self.best_solution = None  # List of Node objects.
        self.best_solution_transitions = None  # Store transitions for elitist update.
        self.global_best_routes = None
        self.fitness_history = []

    def construct_solution(self) -> tuple:
        """
        Constructs a solution for one ant using 3D pheromone.
        Returns:
            transitions (list[tuple]): List of (prev_index, j, decision) tuples.
            final_route (list[Node]): The constructed route (with depot start and end).
        """
        current_location = (self.problem.depot.x, self.problem.depot.y)
        prev_index = 0  # Depot
        unvisited = list(range(1, self.n + 1))
        transitions = []
        final_route = [self.problem.depot]

        while unvisited:
            candidate_options = []
            for j in unvisited:
                customer = self.problem.customers[j - 1]
                ### Determine allowed delivery modes based on customer type.
                if customer.customer_type == 1:
                    allowed = [0] # Only home delivery allowed.
                elif customer.customer_type == 2:
                    allowed = [1] # Only locker delivery allowed.
                else:
                    allowed = [0, 1] # Both home and locker delivery allowed.
                ### Calculate the value for each candidate option.
                for d in allowed:
                    candidate_coord = (customer.x, customer.y) if d == 0 else (customer.assigned_locker.x, customer.assigned_locker.y)
                    distance = euclidean_distance(current_location, candidate_coord)
                    heuristic = 1.0 / distance if distance > 0 else 1e6
                    tau = self.pheromones[prev_index, j, d] ** self.alpha
                    value = tau * (heuristic ** self.beta)
                    candidate_options.append((j, d, value, candidate_coord)) # Destination, delivery mode, value, coordinates.
            total_value = sum(option[2] for option in candidate_options)
            if total_value > 0:
                probs = [option[2] / total_value for option in candidate_options]
            else:
                probs = [1 / len(candidate_options)] * len(candidate_options)
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
            node = self.problem.customers[j - 1] if d == 0 else self.problem.customers[j - 1].assigned_locker
            final_route.append(node)
            current_location = candidate_coord
            prev_index = j
            unvisited.remove(j)
        final_route.append(self.problem.depot)
        return transitions, final_route

    def update_pheromones(self, solutions: list):
        """
        Updates 3D pheromones using the transitions from ant solutions.
        Each solution is a tuple (transitions, fitness).
        """
        # Evaporate pheromones.
        self.pheromones *= (1 - self.evaporation_rate)
        for transitions, fitness in solutions:
            deposit = self.Q / (fitness if fitness > 0 else 1e-8)
            for (i, j, d) in transitions:
                self.pheromones[i, j, d] += deposit

        # Elitist update using best solution transitions.
        if self.best_solution_transitions is not None:
            deposit = self.Q / (self.global_best_fitness if self.global_best_fitness > 0 else 1e-8)
            for (i, j, d) in self.best_solution_transitions:
                self.pheromones[i, j, d] += deposit

        # Enforce fixed settings for customer types.
        for j, customer in enumerate(self.problem.customers, start=1):
            if customer.customer_type == 1:
                self.pheromones[:, j, 1] = 0.0
            elif customer.customer_type == 2:
                self.pheromones[:, j, 0] = 0.0

    def optimize(self, verbose=True):
        """
        Executes the ACO optimization process using 3D pheromone.
        """
        for iteration in range(self.num_iterations):
            solutions = []
            for ant in range(self.num_ants):
                transitions, route = self.construct_solution()
                fitness, routes = self.problem.node2routes(route)
                solutions.append((transitions, fitness))
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.best_solution = route
                    self.best_solution_transitions = transitions
                    self.global_best_routes = routes
            self.update_pheromones(solutions)
            self.fitness_history.append(self.global_best_fitness)
            if verbose:
                print(f"Iteration {iteration + 1}: Best Fitness = {self.global_best_fitness}")
        return self.best_solution, self.global_best_fitness, self.global_best_routes

class PACO(Solver):
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
        pheromones = np.ones((self.n + 1, self.n + 1, 2))
        for j, customer in enumerate(self.problem.customers, start=1):
            if customer.customer_type == 1:
                pheromones[:, j, 1] = 0.0
            elif customer.customer_type == 2:
                pheromones[:, j, 0] = 0.0
        # Create shared memory block for the pheromone matrix
        self.pheromone_shm = shared_memory.SharedMemory(create=True, size=pheromones.nbytes)
        # Create a NumPy array backed by shared memory
        self.shared_pheromones = np.ndarray(pheromones.shape, dtype=pheromones.dtype, buffer=self.pheromone_shm.buf)
        np.copyto(self.shared_pheromones, pheromones)
        self.global_best_fitness = float('inf')
        self.global_best_solution = None
        self.global_best_routes = None
        self.fitness_history = []

    def construct_solution(self, shared_pheromones):
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
                    tau = shared_pheromones[prev_index, j, d]
                    value = (tau ** self.alpha) * (heuristic ** self.beta)
                    candidate_options.append((j, d, value, candidate_coord))

            total_value = sum(option[2] for option in candidate_options)
            probs = [option[2] / total_value for option in candidate_options] if total_value > 0 else [1 / len(candidate_options)] * len(candidate_options)
            r = np.random.random()
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
        self.shared_pheromones *= (1 - self.evaporation_rate)
        for transitions, fitness in solutions:
            deposit = self.Q / (fitness if fitness > 0 else 1e-8)
            for (i, j, d) in transitions:
                self.shared_pheromones[i, j, d] += deposit

        for j, customer in enumerate(self.problem.customers, start=1):
            if customer.customer_type == 1:
                self.shared_pheromones[:, j, 1] = 0.0
            elif customer.customer_type == 2:
                self.shared_pheromones[:, j, 0] = 0.0

    # --- Bundle Multiple Ants in One Task ---
    def multi_ant_solution(self, num_ants_in_task, pheromone_shm_name, shape, dtype):
        # Attach to shared memory for pheromones
        shm = shared_memory.SharedMemory(name=pheromone_shm_name)
        shared_pheromones = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        results = []
        for _ in range(num_ants_in_task):
            worker_ant_start = time.time()
            transitions, final_route = self.construct_solution(shared_pheromones)
            # (Optionally record a mid-time if needed)
            fitness, routes = self.problem.node2routes(final_route)
            worker_ant_end = time.time()
            ant_compute_time = worker_ant_end - worker_ant_start
            # Append result with compute time for overhead estimation
            results.append((transitions, fitness, final_route, routes, ant_compute_time))
        shm.close()  # Detach from shared memory
        return results

    def optimize(self, verbose=True):
        pheromone_shape = self.shared_pheromones.shape
        pheromone_dtype = self.shared_pheromones.dtype
        try:
            with ProcessPoolExecutor() as executor:
                for iteration in range(self.num_iterations):
                    iteration_start = time.time()
                    futures = []
                    submission_times = {}
                    num_tasks = self.num_ants // self.batch_size
                    for _ in range(num_tasks):
                        submit_time = time.time()
                        future = executor.submit(self.multi_ant_solution,
                                                self.batch_size,
                                                self.pheromone_shm.name,
                                                pheromone_shape,
                                                pheromone_dtype)
                        futures.append(future)
                        submission_times[future] = submit_time

                    solutions = []
                    overhead_times = []
                    # Retrieve task results as they complete
                    for future in as_completed(futures):
                        master_end = time.time()
                        round_trip_time = master_end - submission_times[future]
                        task_results = future.result()  # List of ant results from one task
                        total_worker_time = sum(result[4] for result in task_results)
                        # Overhead is the difference between master round-trip and total worker compute time for this task
                        overhead = round_trip_time - total_worker_time
                        overhead_times.append(overhead)
                        solutions.extend(task_results)
                    self.update_pheromones([(transitions, fitness) for (transitions, fitness, _, _, _) in solutions])
                    iteration_end = time.time()
                    avg_overhead_per_task = sum(overhead_times) / len(overhead_times) if overhead_times else 0
                    # Since each task processes self.batch_size ants, compute average overhead per ant:
                    avg_overhead_per_ant = avg_overhead_per_task / self.batch_size
                    # Update pheromones using the computed solutions (ignoring the worker time field)
                    for (transitions, fitness, final_route, routes, _) in solutions:
                        if fitness < self.global_best_fitness:
                            self.global_best_fitness = fitness
                            self.global_best_solution = final_route
                            self.global_best_routes = routes
                    self.fitness_history.append(self.global_best_fitness)
                    if verbose:
                        print(f"Iteration completed {iteration}/{self.num_iterations}\tBest Fitness = {self.global_best_fitness:.2f}\tTime: {iteration_end - iteration_start:.2f}")
        finally:
            self.cleanup()
        return self.global_best_solution, self.global_best_fitness, self.global_best_routes

    # --- Cleanup Shared Memory ---
    def cleanup(self):
        self.pheromone_shm.close()
        self.pheromone_shm.unlink()

def main_PACO():
    instance = Problem()
    instance.load_data("data/25/C101_co_25.txt")
    # aco = SACO(instance, num_ants=1000, num_iterations=100, alpha=1.0, beta=1.0, evaporation_rate=0.1, Q=1.0)
    aco = PACO(instance, num_ants=5000, batch_size=100, num_iterations=100, alpha=1.0, beta=1.0, evaporation_rate=0.1, Q=1.0)
    import timeit
    export_pheromones_heatmap(aco.shared_pheromones, filename="output/initial_pheromones.png")
    run_time = timeit.timeit(lambda: aco.optimize(verbose=True), number=1)
    print(f"Best Fitness: {aco.global_best_fitness}")
    print(f"Execution Time: {run_time:.2f} seconds")
    print("Best Solution:")
    print_routes(aco.global_best_routes)
    aco.plot_fitness_history()
    aco.plot_routes()

if __name__ == "__main__":
    main_PACO()