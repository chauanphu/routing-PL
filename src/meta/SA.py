import random
import math
from solver import Node, Problem, Solver, print_routes

class SimulatedAnnealing(Solver):
    def __init__(self, problem: Problem, init_temperature: float = 1000.0, cooling_rate: float = 0.995, beta: float = 1.0,
                 min_temperature: float = 0.5, iterations_per_temp: int = 100, max_iters: int = 100, non_improvement: int = 100):
        """
        Initialize the Simulated Annealing optimizer.
        
        Parameters:
            problem: an instance of the Problem class containing VRP data and the evaluate() method.
            init_temperature (float): starting temperature.
            cooling_rate (float): factor by which temperature is multiplied each cooling step.
            min_temperature (float): stopping temperature.
            iterations_per_temp (int): number of candidate moves per temperature.
        """
        super().__init__()
        self.problem = problem
        self.T = init_temperature
        self.cooling_rate = cooling_rate
        self.beta = beta
        self.min_temperature = min_temperature
        self.iterations_per_temp = iterations_per_temp
        self.max_iters = max_iters
        self.non_improvement = non_improvement
        self.num_iterations = 0

    def neighbor(self, solution: list[Node]) -> list[Node]:
        """
        Generate a neighbor solution by perturbing one or more positions.
        Three types of neighborhood moves are used:
        - Swap: Two elements are exchanged.
        - Insertion: An element is removed from one position and inserted at another.
        - Inversion: A sublist of the solution is reversed.
        
        The move is chosen randomly:
        if random < 1/3: swap,
        if 1/3 <= random < 2/3: insertion,
        else: inversion.
        
        Parameters:
            solution (list[float]): The current solution (continuous values).
        
        Returns:
            neighbor_solution (list[float]): A new solution after applying a perturbation.
        """
        # Copy the solution to avoid modifying the original.
        neighbor_solution = solution.copy()
        
        # Determine the range for customer indices (exclude first and last positions).
        n = len(neighbor_solution)
        if n <= 2:
            return neighbor_solution  # Nothing to change if there are no customer positions.
        
        customer_indices = list(range(1, n - 1))
        move_choice = random.random()  # random float in [0, 1)
        
        if move_choice < 1/3:
            # Swap: randomly select two distinct customer indices and swap their values.
            i, j = random.sample(customer_indices, 2)
            neighbor_solution[i], neighbor_solution[j] = neighbor_solution[j], neighbor_solution[i]
        
        elif move_choice < 2/3:
            # Insertion: remove an element from a random customer position and insert it at another random position.
            i = random.choice(customer_indices)
            element = neighbor_solution.pop(i)
            # Adjust the available indices after removal.
            new_customer_indices = list(range(1, n - 1))
            # Choose a new position from the new indices.
            j = random.choice(new_customer_indices)
            neighbor_solution.insert(j, element)
        
        else:
            # Inversion: choose two customer indices and reverse the sublist between them (inclusive).
            i, j = sorted(random.sample(customer_indices, 2))
            neighbor_solution[i:j+1] = neighbor_solution[i:j+1][::-1]
        
        return neighbor_solution
    
    def optimize(self, verbose=True) -> tuple[list[float], float, list[list]]:
        """
        Run the Simulated Annealing optimization.
        
        Returns:
            best_solution (list[float]): the continuous vector representing the best solution.
            best_cost (float): the best objective value (total travel distance).
            best_routes (list[list[Node]]): the corresponding list of routes (each route is a list of Node objects).
        """
        # Generate an initial solution.
        # 1. Generate a random solution.
        current_solution = self.problem.random_route()
        current_cost, current_routes = self.problem.permu2route(current_solution)
        # 2. Initialize best solution with the random solution.
        best_solution = current_solution
        best_cost = current_cost
        best_routes = current_routes
        # 3. Print initial solution.
        # Main SA loop.
        iter_count = 0
        non_improvement_count = 0
        found_best = False
        while self.T > self.min_temperature and iter_count < self.max_iters and non_improvement_count < self.non_improvement:
            iter_count += 1
            print(f"Iteraion: {iter_count}/{self.max_iters}:\tTemperature: {self.T:.2f} / {self.min_temperature:.2f}, Best Objective: {best_cost:.2f}")
            # print(f"Temperature: {self.T:.2f} / {self.min_temperature:.2f}, Best Objective: {best_cost:.2f}")
            for _ in range(self.iterations_per_temp):
                    # Generate a neighboring solution.
                candidate_solution = self.neighbor(current_solution)
                candidate_cost, candidate_routes = self.problem.permu2route(candidate_solution)
                delta = candidate_cost - current_cost
                
                # Accept the candidate if it's better, or with a probability if worse.
                if delta < 0 or random.random() < math.exp(-delta / (self.beta * self.T)):
                    current_solution = candidate_solution
                    current_cost = candidate_cost
                    current_routes = candidate_routes
                    # Update best solution if improvement is found.
                    if candidate_cost < best_cost:
                        best_solution = candidate_solution
                        best_cost = candidate_cost
                        best_routes = candidate_routes
                        non_improvement_count = 0
                        found_best = True
            
            # Cool down the temperature.
            self.T *= self.cooling_rate
            if not found_best:
                non_improvement_count += 1
                found_best = True
            self.fitness_history.append(best_cost)

        self.global_best_position = best_solution
        self.global_best_fitness = best_cost
        self.num_iterations = iter_count

        return best_solution, best_cost
    
if __name__ == "__main__":
    # Assuming you have already created and loaded your Problem instance.
    problem = Problem()
    problem.load_data("data/25/C101_co_25.txt")  # Make sure your data file is correctly formatted.
    
    # Create an instance of the Simulated Annealing optimizer.
    sa = SimulatedAnnealing(problem, init_temperature=10.0, cooling_rate=0.97, beta=1.0,
                            min_temperature=0.1, iterations_per_temp=600, max_iters=200, non_improvement=50)
    
    # Run the optimization.
    sa.optimize()
    best_cost, best_solution = problem.permu2route(sa.global_best_position)
    print("Distance: ", best_cost)
    print_routes(best_solution)
    sa.plot_fitness_history()