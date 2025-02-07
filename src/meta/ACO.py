import math
import random
import numpy as np
if __name__ == '__main__':
    from solver import Problem, Solver, Node
else:
    from meta.solver import Problem, Solver, Node

def euclidean_distance(p1, p2):
    """Compute Euclidean distance between two 2D points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

class AntColonyOptimization(Solver):
    def __init__(self, problem: Problem, num_ants=20, num_iterations=100,
                 alpha=1.0, beta=2.0, evaporation_rate=0.1, Q=1.0):
        """
        Initializes the ACO optimizer for VRPPL.
        
        The candidate set for ordering includes the depot (index 0) and all customers (indices 1..n)
        with representative coordinates defined as:
          - Type I: customer's house coordinates.
          - Type II: assigned locker coordinates.
          - Type III: average of house and locker coordinates.
        
        For type III customers the final delivery decision (home vs. locker) is determined by a
        secondary pheromone process.
        
        Parameters:
            problem (Problem): The VRP problem instance.
            num_ants (int): Number of ants per iteration.
            num_iterations (int): Total number of iterations.
            alpha (float): Relative importance of pheromone (ordering).
            beta (float): Relative importance of heuristic information.
            evaporation_rate (float): Rate at which pheromone evaporates.
            Q (float): Constant for pheromone deposit.
        """
        # The objective function is provided by problem.permu2route.
        super().__init__(problem.node2routes, num_iterations=num_iterations)
        self.problem: Problem = problem
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.Q = Q

        # Number of customers.
        self.n = self.problem.num_customers

        # --- Build candidate set for ordering ---
        # Depot coordinates.
        self.depot_coord = (self.problem.depot.x, self.problem.depot.y)
        # For each customer, compute a representative coordinate.
        # Type I: use house; Type II: use locker; Type III: use average.
        self.candidate_coords = [self.depot_coord]  # index 0 is depot.
        for customer in self.problem.customers:
            if customer.customer_type == 1:
                rep = (customer.x, customer.y)
            elif customer.customer_type == 2:
                rep = (customer.assigned_locker.x, customer.assigned_locker.y)
            elif customer.customer_type == 3:
                rep = ((customer.x + customer.assigned_locker.x) / 2,
                       (customer.y + customer.assigned_locker.y) / 2)
            else:
                rep = (customer.x, customer.y)
            self.candidate_coords.append(rep)

        # Initialize ordering pheromone matrix over indices 0..n.
        self.pheromones = np.ones((self.n + 1, self.n + 1))
        # Build heuristic matrix: η[i][j] = 1 / distance(candidate_coords[i], candidate_coords[j])
        self.eta = np.zeros((self.n + 1, self.n + 1))
        for i in range(self.n + 1):
            for j in range(self.n + 1):
                if i == j:
                    self.eta[i][j] = 0.0
                else:
                    d = euclidean_distance(self.candidate_coords[i], self.candidate_coords[j])
                    self.eta[i][j] = 1.0 / d if d != 0 else 1e6

        # --- Initialize type III decision pheromones ---
        # For each type III customer, maintain a dictionary with keys "locker" and "home".
        self.type3_pheromones = {}
        for customer in self.problem.customers:
            if customer.customer_type == 3:
                self.type3_pheromones[customer.node_id] = {"locker": 1.0, "home": 1.0}

        self.global_best_fitness = float('inf')
        self.global_best_solution = None  # Final route (list of Node objects)
        self.global_best_routes = None
        self.fitness_history = []

    def construct_solution(self):
        """
        Constructs one ant's solution.
        
        First, the ant builds an ordering (a permutation over candidate indices 0..n)
        using roulette-wheel selection based on the ordering pheromones and heuristic values.
        
        Then, the final route is built by traversing the ordering (ignoring the depot markers)
        and, for each customer:
          - Type I: use the customer’s house.
          - Type II: use the assigned locker.
          - Type III: use the type III decision pheromone to choose between locker and home.
        
        Returns:
            ordering (list of int): The permutation over candidate indices (with depot as 0 at start/end).
            final_route (list of Node): The route as a list of Node objects (starting and ending with depot).
            type3_decisions (dict): A mapping from type III customer node_id to the decision chosen ("locker" or "home").
        """
        # --- Construct ordering ---
        current = 0  # start at depot (index 0)
        unvisited = list(range(1, self.n + 1))  # candidate indices for customers
        ordering = [0]
        while unvisited:
            probs = []
            for j in unvisited:
                tau = self.pheromones[current][j] ** self.alpha
                eta_val = self.eta[current][j] ** self.beta
                probs.append(tau * eta_val)
            total = sum(probs)
            if total == 0:
                probs = [1/len(unvisited)] * len(unvisited)
            else:
                probs = [p/total for p in probs]
            r = random.random()
            cumulative = 0.0
            for idx, j in enumerate(unvisited):
                cumulative += probs[idx]
                if r <= cumulative:
                    next_index = j
                    break
            ordering.append(next_index)
            unvisited.remove(next_index)
            current = next_index
        ordering.append(0)  # return to depot

        # --- Build final route and record type III decisions ---
        final_route = [self.problem.depot]
        type3_decisions = {}  # key: customer.node_id, value: "locker" or "home"
        # For each candidate index in the ordering (skipping the depot markers)
        for idx in ordering[1:-1]:
            customer = self.problem.customers[idx - 1]  # candidate index i corresponds to customer i-1
            if customer.customer_type == 1:
                # Type I: always home delivery.
                final_route.append(customer)
            elif customer.customer_type == 2:
                # Type II: always locker delivery.
                final_route.append(customer.assigned_locker)
            elif customer.customer_type == 3:
                # Type III: decide based on pheromone values.
                ph_values = self.type3_pheromones[customer.node_id]
                p_lock = ph_values["locker"] / (ph_values["locker"] + ph_values["home"])
                if random.random() < p_lock:
                    final_route.append(customer.assigned_locker)
                    type3_decisions[customer.node_id] = "locker"
                else:
                    final_route.append(customer)
                    type3_decisions[customer.node_id] = "home"
            else:
                # Fallback: use customer's house.
                final_route.append(customer)
        final_route.append(self.problem.depot)
        return ordering, final_route, type3_decisions

    def update_pheromones(self, all_solutions):
        """
        Updates both the ordering pheromone matrix and the type III decision pheromones.
        
        Parameters:
            all_solutions (list of tuples): Each tuple is (ordering, fitness, type3_decisions).
            The ordering is a list of candidate indices (with depot markers),
            and type3_decisions is a dict mapping type III customer node_id to decision.
        """
        # --- Update ordering pheromones ---
        self.pheromones *= (1 - self.evaporation_rate)
        for ordering, fitness, _ in all_solutions:
            deposit = self.Q / (fitness if fitness > 0 else 1e-8)
            for i in range(len(ordering) - 1):
                a = ordering[i]
                b = ordering[i + 1]
                self.pheromones[a][b] += deposit
                self.pheromones[b][a] += deposit

        # --- Update type III decision pheromones ---
        # Accumulate deposits for each type III customer.
        type3_deposits = {}
        # Initialize deposit accumulators.
        for customer in self.problem.customers:
            if customer.customer_type == 3:
                type3_deposits[customer.node_id] = {"locker": 0.0, "home": 0.0}
        # For each ant solution, add deposit for the chosen decision.
        for _, fitness, decisions in all_solutions:
            deposit = self.Q / (fitness if fitness > 0 else 1e-8)
            for cust_id, decision in decisions.items():
                type3_deposits[cust_id][decision] += deposit
        # Now update the type III pheromone values with evaporation.
        for cust_id, deposits in type3_deposits.items():
            self.type3_pheromones[cust_id]["locker"] = ((1 - self.evaporation_rate) *
                                                          self.type3_pheromones[cust_id]["locker"] +
                                                          deposits["locker"])
            self.type3_pheromones[cust_id]["home"] = ((1 - self.evaporation_rate) *
                                                      self.type3_pheromones[cust_id]["home"] +
                                                      deposits["home"])

    def optimize(self, verbose=True):
        """
        Executes the ACO optimization process.
        
        Returns:
            best_solution (list of Node): The best final route found.
            best_fitness (float): The fitness value of the best route.
            best_routes: Additional route information from the evaluation method.
        """
        for iteration in range(self.num_iterations):
            all_solutions = []
            for ant in range(self.num_ants):
                ordering, final_route, type3_decisions = self.construct_solution()
                fitness, routes = self.problem.node2routes(final_route)
                all_solutions.append((ordering, fitness, type3_decisions))
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_solution = final_route
                    self.global_best_routes = routes
            self.update_pheromones(all_solutions)
            self.fitness_history.append(self.global_best_fitness)
            if verbose:
                print(f"Iteration {iteration + 1}: Best Fitness = {self.global_best_fitness}")
        return self.global_best_solution, self.global_best_fitness, self.global_best_routes

if __name__ == '__main__':
    import time
    from solver import Node, Problem, print_routes
    # Create a problem instance
    instance = Problem()
    instance.load_data("data/25/C101_co_25.txt")

    start_time = time.time()
    # Load the solver
    aco = AntColonyOptimization(problem=instance, num_ants=1000, num_iterations=100)
    aco.optimize()
    print("Distance: ", aco.global_best_fitness)
    print([node.node_id for node in aco.global_best_solution])
    print_routes(aco.global_best_routes)
    print("Elapsed time (s):", time.time() - start_time)
    aco.plot_fitness_history()
    aco.plot_routes()