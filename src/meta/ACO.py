import random
import numpy as np
if __name__ == '__main__':
    from solver import Problem, Solver, Node
else:
    from meta.solver import Problem, Solver, Node

class AntColonyOptimization(Solver):
    def __init__(self, problem: Problem, num_ants=20, num_iterations=100,
                 alpha=1.0, beta=2.0, evaporation_rate=0.1, Q=1.0):
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
            initial_solution = self.problem.initialize_solution(p=0.1)[1]
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

if __name__ == '__main__':
    import time
    from solver import Node, Problem, print_routes
    # Create a problem instance
    instance = Problem()
    instance.load_data("data/100/C102_co_100.txt")

    start_time = time.time()
    # Load the solver
    aco = AntColonyOptimization(problem=instance, num_ants=1000, num_iterations=100)
    aco.optimize()
    print("Distance: ", aco.global_best_fitness)
    print([node.node_id for node in aco.best_solution])
    print_routes(aco.global_best_routes)
    print("Elapsed time (s):", time.time() - start_time)
    aco.plot_fitness_history()
    aco.plot_routes()