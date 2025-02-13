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
        Initializes the ACO optimizer for VRPPL using a single 3D pheromone matrix.
        
        The 3D pheromone matrix has dimensions (n+1) x (n+1) x 2, where n is the number of customers.
        The first index represents the "from" node (with 0 representing the depot and 1..n representing customers),
        the second index represents the "to" customer (always in 1..n), and the third dimension represents the 
        delivery decision: index 0 for home delivery and index 1 for locker delivery.
        
        For each customer:
          - Type I: only home delivery is allowed (locker branch is forced to 0).
          - Type II: only locker delivery is allowed (home branch is forced to 0).
          - Type III: both options are learnable.
        
        In a solution construction, when moving from a current node (starting at the depot) to an unvisited customer j,
        the ant considers each allowed delivery mode d. The probability of selecting candidate j with decision d is
        proportional to:
             (pheromone[from, j, d]^alpha) * ( (1/distance)^beta )
        where distance is computed between the current location and the candidate's location (customer's house if d==0,
        or assigned locker if d==1).
        
        After selecting a candidate, the composite decision (j, d) is appended to the route. For example, if from depot 0
        an ant selects customer 7 with decision 1 (locker), then the ant will visit customer 7’s assigned locker.
        
        Parameters:
            problem (Problem): The VRP problem instance.
            num_ants (int): Number of ants per iteration.
            num_iterations (int): Total number of iterations.
            alpha (float): Relative importance of pheromone.
            beta (float): Relative importance of heuristic (inverse distance).
            evaporation_rate (float): Rate at which pheromone evaporates.
            Q (float): Constant used for pheromone deposit.
        """
        # The objective function is provided by problem.node2routes.
        super().__init__(problem.node2routes, num_iterations=num_iterations)
        self.problem: Problem = problem
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.Q = Q

        # n: number of customers
        self.n = self.problem.num_customers

        # We'll use indices 0..n where 0 is the depot and 1..n refer to customers.
        # Initialize a single 3D pheromone matrix with ones.
        # Dimensions: (n+1) x (n+1) x 2.
        # Note: For transitions into a customer j, only one decision is allowed for type I (home) and type II (locker).
        self.pheromones = np.ones((self.n + 1, self.n + 1, 2))
        for j, customer in enumerate(self.problem.customers, start=1):
            if customer.customer_type == 1:
                # For type I, disallow locker delivery (index 1) by setting pheromone to 0.
                self.pheromones[:, j, 1] = 0.0
            elif customer.customer_type == 2:
                # For type II, disallow home delivery (index 0) by setting pheromone to 0.
                self.pheromones[:, j, 0] = 0.0
            # For type III, both indices remain learnable (initialized to 1).

        # Global best solution tracking.
        self.global_best_fitness = float('inf')
        self.global_best_solution = None  # Final route (list of Node objects)
        self.global_best_routes = None
        self.fitness_history = []

    def construct_solution(self):
        """
        Constructs one ant's solution using the integrated 3D pheromone matrix.
        
        Starting at the depot (index 0), the ant builds a route by iteratively selecting an unvisited customer j
        along with a delivery decision d (0 for home, 1 for locker) according to a probability distribution.
        
        For each unvisited customer j, if customer j is:
          - Type I: only decision 0 (home) is allowed.
          - Type II: only decision 1 (locker) is allowed.
          - Type III: both decisions are considered.
        
        The probability of selecting (j, d) is proportional to:
             (pheromones[prev, j, d]^alpha) * ( (1/distance)^beta )
        where:
             - prev is the index of the last visited customer (or 0 for the depot).
             - distance is computed between the current location (a 2D coordinate) and the candidate's coordinate,
               which is (customer.x, customer.y) if d==0 (home) or (customer.assigned_locker.x, customer.assigned_locker.y)
               if d==1 (locker).
        
        Returns:
            transitions (list of tuples): Each tuple is (i, j, d) representing a transition from node index i
                                          to customer j with decision d.
            final_route (list of Node): The constructed route as a list of Node objects (starting and ending with depot).
        """
        # Start at depot.
        current_location = (self.problem.depot.x, self.problem.depot.y)
        prev_index = 0  # 0 represents depot.
        unvisited = list(range(1, self.n + 1))  # Candidate customer indices.
        transitions = []  # List of transitions: (prev_index, j, d)
        final_route = [self.problem.depot]

        while unvisited:
            candidate_options = []  # List of tuples: (j, d, combined_value, candidate_coord)
            for j in unvisited:
                customer = self.problem.customers[j - 1]
                # Determine allowed decisions based on customer type.
                if customer.customer_type == 1:
                    allowed = [0]
                elif customer.customer_type == 2:
                    allowed = [1]
                elif customer.customer_type == 3:
                    allowed = [0, 1]
                else:
                    allowed = [0]  # Fallback
                
                for d in allowed:
                    # Determine candidate coordinate based on decision.
                    if d == 0:
                        candidate_coord = (customer.x, customer.y)
                    else:  # d == 1
                        candidate_coord = (customer.assigned_locker.x, customer.assigned_locker.y)
                    
                    distance = euclidean_distance(current_location, candidate_coord)
                    # Avoid division by zero.
                    heuristic = 1.0 / distance if distance > 0 else 1e6
                    tau = self.pheromones[prev_index, j, d]
                    value = (tau ** self.alpha) * (heuristic ** self.beta)
                    candidate_options.append((j, d, value, candidate_coord))
            
            total_value = sum(option[2] for option in candidate_options)
            # Normalize probabilities.
            if total_value == 0:
                probs = [1 / len(candidate_options)] * len(candidate_options)
            else:
                probs = [option[2] / total_value for option in candidate_options]
            
            # Roulette-wheel selection.
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
            # Append the proper node to the final route.
            customer = self.problem.customers[j - 1]
            if d == 0:
                final_route.append(customer)
            else:
                final_route.append(customer.assigned_locker)
            # Update state.
            current_location = candidate_coord
            prev_index = j
            unvisited.remove(j)

        final_route.append(self.problem.depot)
        return transitions, final_route

    def update_pheromones(self, solutions):
        """
        Updates the integrated 3D pheromone matrix based on the ant solutions.
        
        For each solution (defined by its sequence of transitions and fitness), pheromone is deposited on the
        corresponding transitions (i, j, d) in the pheromone matrix. Then, evaporation is applied to all entries.
        
        Parameters:
            solutions (list of tuples): Each tuple is (transitions, fitness) where:
                - transitions: a list of (i, j, d) transitions from the constructed solution.
                - fitness: the fitness value of the solution.
        """
        # Evaporation.
        self.pheromones *= (1 - self.evaporation_rate)
        for transitions, fitness in solutions:
            deposit = self.Q / (fitness if fitness > 0 else 1e-8)
            for (i, j, d) in transitions:
                self.pheromones[i, j, d] += deposit

        # Enforce hard constraints for disallowed decisions.
        for j, customer in enumerate(self.problem.customers, start=1):
            if customer.customer_type == 1:
                self.pheromones[:, j, 1] = 0.0
            elif customer.customer_type == 2:
                self.pheromones[:, j, 0] = 0.0

    def optimize(self, verbose=True):
        """
        Executes the ACO optimization process with the integrated 3D pheromone matrix.
        
        For each iteration, a number of ants construct solutions. Each solution is evaluated using
        problem.node2routes, and pheromone updates are applied. The best solution found is tracked.
        
        Returns:
            best_solution (list of Node): The best final route found.
            best_fitness (float): The fitness value of the best route.
            best_routes: Additional route information from the evaluation method.
        """
        for iteration in range(self.num_iterations):
            solutions = []  # List of tuples: (transitions, fitness)
            for ant in range(self.num_ants):
                transitions, final_route = self.construct_solution()
                fitness, routes = self.problem.node2routes(final_route)
                solutions.append((transitions, fitness))
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_solution = final_route
                    self.global_best_routes = routes
            self.update_pheromones(solutions)
            self.fitness_history.append(self.global_best_fitness)
            if verbose:
                print(f"Iteration {iteration + 1}: Best Fitness = {self.global_best_fitness}")
        return self.global_best_solution, self.global_best_fitness, self.global_best_routes


if __name__ == '__main__':
    import time
    from solver import Node, Problem, print_routes
    # Create a problem instance
    instance = Problem()
    instance.load_data("data/100/C101_co_100.txt")

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