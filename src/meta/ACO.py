import asyncio
from concurrent.futures import ProcessPoolExecutor
import math
import random
import numpy as np

if __name__ == "__main__":
    from solver import Problem, Solver
else:
    from meta.solver import Problem, Solver

def euclidean_distance(p1, p2):
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
    
class Colony:
    """
    A helper class representing an independent ACO colony.
    Each colony maintains its own (local) pheromone matrix and runs for a number of iterations.
    """
    def __init__(self, colony_id, problem, num_iterations, sync_interval,
                 num_ants, alpha, beta, evaporation_rate, Q, initial_pheromones):
        self.colony_id = colony_id
        self.problem = problem
        self.num_iterations = num_iterations
        self.sync_interval = sync_interval  # How often to synchronize with the global pheromone matrix
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.Q = Q
        # Start with a local copy of the global pheromone matrix.
        self.pheromones = np.copy(initial_pheromones)
        self.local_best_fitness = float('inf')
        self.local_best_solution = None

    def construct_solution(self):
        """
        Constructs one ant's solution using the local 3D pheromone matrix.
        Returns:
            transitions: list of (prev_index, j, decision) tuples.
            final_route: list of Node objects representing the constructed route.
        """
        current_location = (self.problem.depot.x, self.problem.depot.y)
        prev_index = 0
        unvisited = list(range(1, self.problem.num_customers + 1))
        transitions = []
        final_route = [self.problem.depot]

        while unvisited:
            candidate_options = []
            for j in unvisited:
                customer = self.problem.customers[j - 1]
                # Determine allowed delivery modes based on customer type.
                if customer.customer_type == 1:
                    allowed = [0]
                elif customer.customer_type == 2:
                    allowed = [1]
                else:  # Flexible customer (type 3)
                    allowed = [0, 1]
                for d in allowed:
                    # Use home coordinates if d==0, locker coordinates if d==1.
                    candidate_coord = (customer.x, customer.y) if d == 0 else (customer.assigned_locker.x, customer.assigned_locker.y)
                    distance = euclidean_distance(current_location, candidate_coord)
                    heuristic = 1.0 / distance if distance > 0 else 1e6
                    tau = self.pheromones[prev_index, j, d]
                    value = (tau ** self.alpha) * (heuristic ** self.beta)
                    candidate_options.append((j, d, value, candidate_coord))
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
            # Append the chosen node: home (if d==0) or the assigned locker (if d==1).
            if d == 0:
                final_route.append(self.problem.customers[j - 1])
            else:
                final_route.append(self.problem.customers[j - 1].assigned_locker)
            current_location = candidate_coord
            prev_index = j
            unvisited.remove(j)
        final_route.append(self.problem.depot)
        return transitions, final_route

    def update_local_pheromones(self, solutions):
        """Local pheromone update for the colony."""
        # Evaporate pheromones.
        self.pheromones *= (1 - self.evaporation_rate)
        for transitions, fitness in solutions:
            deposit = self.Q / (fitness if fitness > 0 else 1e-8)
            for (i, j, d) in transitions:
                self.pheromones[i, j, d] += deposit
        # Enforce the fixed pheromone settings for customer types.
        for j, customer in enumerate(self.problem.customers, start=1):
            if customer.customer_type == 1:
                self.pheromones[:, j, 1] = 0.0
            elif customer.customer_type == 2:
                self.pheromones[:, j, 0] = 0.0

    def single_ant_solution(self):
        """
        Constructs one ant's solution and evaluates its fitness.
        Returns:
            (transitions, fitness)
        """
        transitions, final_route = self.construct_solution()
        fitness, _ = self.problem.node2routes(final_route)
        return transitions, fitness

    async def run(self, global_pheromone_queue):
        print(f"[Colony {self.colony_id}] Starting optimization for {self.num_iterations} iterations.")
        for iteration in range(self.num_iterations):
            solutions = [self.single_ant_solution() for _ in range(self.num_ants)]
            self.update_local_pheromones(solutions)
            # Update local best.
            for transitions, fitness in solutions:
                if fitness < self.local_best_fitness:
                    self.local_best_fitness = fitness

            # Yield control to allow other colonies to run.
            await asyncio.sleep(0)
            
            if iteration % 10 == 0:
                print(f"[Colony {self.colony_id}] Iteration {iteration+1}/{self.num_iterations} complete. Local best fitness: {self.local_best_fitness}")
            if iteration % self.sync_interval == 0:
                print(f"[Colony {self.colony_id}] Pushing pheromone update at iteration {iteration+1}.")
                await global_pheromone_queue.put((self.colony_id, np.copy(self.pheromones)))
        print(f"[Colony {self.colony_id}] Optimization complete. Final local best fitness: {self.local_best_fitness}")
        return self.local_best_fitness

async def global_pheromone_manager(global_pheromones, num_colonies, global_pheromone_queue, sync_interval):
    """
    Aggregates updates from colonies and updates the global pheromone matrix periodically.
    """
    print("[Global Manager] Starting global pheromone management.")
    iteration = 0
    while True:
        updates = []
        # Try to gather updates from all colonies with a timeout to prevent freezing.
        for _ in range(num_colonies):
            try:
                update = await asyncio.wait_for(global_pheromone_queue.get(), timeout=30)
                updates.append(update[1])
                print(f"[Global Manager] Received update from Colony {update[0]}.")
            except asyncio.TimeoutError:
                print("[Global Manager] Timeout waiting for colony update. Proceeding with available updates.")
                break
        if updates:
            aggregated = np.mean(updates, axis=0)
            global_pheromones[:] = (global_pheromones + aggregated) / 2
            print(f"[Global Manager] Aggregated pheromone update at sync iteration {iteration+1}.")
        else:
            print("[Global Manager] No updates received in this cycle.")
        iteration += 1
        await asyncio.sleep(sync_interval)

async def run_multicolony(problem, num_colonies, num_iterations, num_ants, sync_interval,
                           alpha, beta, evaporation_rate, Q, global_pheromones):
    """
    Launches multiple colonies asynchronously and returns the best fitness found and the final global pheromones.
    """
    print(f"[Multicolony] Starting optimization with {num_colonies} colonies, each running {num_iterations} iterations.")
    global_pheromone_queue = asyncio.Queue()
    colonies = [
        Colony(
            colony_id=i,
            problem=problem,
            num_iterations=num_iterations,
            sync_interval=sync_interval,
            num_ants=num_ants,
            alpha=alpha,
            beta=beta,
            evaporation_rate=evaporation_rate,
            Q=Q,
            initial_pheromones=global_pheromones
        )
        for i in range(num_colonies)
    ]
    colony_tasks = [asyncio.create_task(colony.run(global_pheromone_queue)) for colony in colonies]
    manager_task = asyncio.create_task(
        global_pheromone_manager(global_pheromones, num_colonies, global_pheromone_queue, sync_interval)
    )
    # Wait for all colonies to finish.
    results = await asyncio.gather(*colony_tasks)
    # Cancel the manager task once colonies have finished.
    manager_task.cancel()
    best_overall = min(results)
    print(f"[Multicolony] All colonies complete. Best overall fitness: {best_overall}")
    return best_overall, global_pheromones

class AsyncMulticolonyACO(Solver):
    """
    An asynchronous, multicolony ACO solver.
    Each colony runs independently and periodically updates a shared global pheromone matrix.
    """
    def __init__(self, problem, num_colonies=4, num_iterations=100, num_ants=100, sync_interval=10,
                 alpha=1.0, beta=2.0, evaporation_rate=0.1, Q=1.0):
        """
        Initializes the asynchronous multicolony ACO solver.
        
        Parameters:
            problem (Problem): The routing problem instance.
            num_colonies (int): Number of independent colonies.
            num_iterations (int): Iterations per colony.
            num_ants (int): Number of ants per colony per iteration.
            sync_interval (int): Frequency (in iterations) at which colonies push updates.
            alpha (float): Influence of pheromone.
            beta (float): Influence of heuristic.
            evaporation_rate (float): Pheromone evaporation rate.
            Q (float): Constant used for pheromone deposit.
        """
        super().__init__(problem.node2routes, num_iterations=num_iterations)
        self.problem = problem
        self.num_colonies = num_colonies
        self.num_iterations = num_iterations
        self.num_ants = num_ants
        self.sync_interval = sync_interval
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.Q = Q
        self.n = self.problem.num_customers

        # Initialize the global pheromone matrix.
        self.global_pheromones = np.ones((self.n + 1, self.n + 1, 2))
        for j, customer in enumerate(self.problem.customers, start=1):
            if customer.customer_type == 1:
                self.global_pheromones[:, j, 1] = 0.0
            elif customer.customer_type == 2:
                self.global_pheromones[:, j, 0] = 0.0

        self.global_best_fitness = float('inf')
        self.global_best_solution = None  # Extend this if you want to store routes/solutions.
        self.fitness_history = []

    def optimize(self, verbose=True):
        """
        Runs the asynchronous multicolony ACO optimization and returns the best solution found.
        """
        if verbose:
            print("[AsyncMulticolonyACO] Starting optimization.")
        best_fitness, final_pheromones = asyncio.run(
            run_multicolony(
                self.problem,
                num_colonies=self.num_colonies,
                num_iterations=self.num_iterations,
                num_ants=self.num_ants,
                sync_interval=self.sync_interval,
                alpha=self.alpha,
                beta=self.beta,
                evaporation_rate=self.evaporation_rate,
                Q=self.Q,
                global_pheromones=self.global_pheromones
            )
        )
        self.global_best_fitness = best_fitness
        self.fitness_history.append(best_fitness)
        if verbose:
            print("[AsyncMulticolonyACO] Optimization complete. Best fitness =", best_fitness)
        # Return best solution (if stored) and best fitness.
        return self.global_best_solution, self.global_best_fitness

if __name__ == '__main__':
    import time
    from solver import Node, Problem, print_routes
    # Create a problem instance
    instance = Problem()
    instance.load_data("data/25/C101_co_25.txt")

    start_time = time.time()
    # Load the solver
    aco = AntColonyOptimization(instance, num_iterations=50, num_ants=1000, batch_size=100,
                                  alpha=1.0, beta=2.0, evaporation_rate=0.1, Q=1.0)
    aco.optimize()
    print("Elapsed time (s):", time.time() - start_time)
    print("Distance: ", aco.global_best_fitness)
    print([node.node_id for node in aco.global_best_solution])
    print_routes(aco.global_best_routes)
    aco.plot_fitness_history()
    aco.plot_routes()