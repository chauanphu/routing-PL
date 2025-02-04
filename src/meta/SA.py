import random
import math
from solver import Problem

class SimulatedAnnealing:
    def __init__(self, problem: Problem, init_temperature: float = 1000.0, cooling_rate: float = 0.995, 
                 min_temperature: float = 0.5, iterations_per_temp: int = 100):
        """
        Initialize the Simulated Annealing optimizer.
        
        Parameters:
            problem: an instance of the Problem class containing VRP data and the evaluate() method.
            init_temperature (float): starting temperature.
            cooling_rate (float): factor by which temperature is multiplied each cooling step.
            min_temperature (float): stopping temperature.
            iterations_per_temp (int): number of candidate moves per temperature.
        """
        self.problem = problem
        self.T = init_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.iterations_per_temp = iterations_per_temp

    def random_solution(self) -> list[float]:
        """
        Generate a random continuous solution.
        Each customer is assigned a random value in [0,1].
        """
        n = self.problem.num_customers
        return [random.random() for _ in range(n)]
    
    def neighbor(self, solution: list[float]) -> list[float]:
        """
        Generate a neighbor solution by perturbing one or more positions.
        Here, we perturb one randomly chosen element by adding a small Gaussian noise.
        """
        neighbor_solution = solution.copy()
        # Choose a random index to perturb.
        idx = random.randrange(len(solution))
        # Perturb using Gaussian noise.
        perturbation = random.gauss(0, 0.1)
        neighbor_solution[idx] += perturbation
        # Optionally clip the value to keep it in [0,1].
        neighbor_solution[idx] = max(0.0, min(1.0, neighbor_solution[idx]))
        return neighbor_solution
    
    def run(self) -> tuple[list[float], float, list[list]]:
        """
        Run the Simulated Annealing optimization.
        
        Returns:
            best_solution (list[float]): the continuous vector representing the best solution.
            best_cost (float): the best objective value (total travel distance).
            best_routes (list[list[Node]]): the corresponding list of routes (each route is a list of Node objects).
        """
        # Generate an initial solution.
        current_solution = self.random_solution()
        current_cost, current_routes = self.problem.to_route(current_solution)
        best_solution = current_solution
        best_cost = current_cost
        best_routes = current_routes

        # Main SA loop.
        while self.T > self.min_temperature:
            print(f"Temperature: {self.T:.2f} / {self.min_temperature:.2f}, Best Objective: {best_cost:.2f}")
            for _ in range(self.iterations_per_temp):
                # Generate a neighboring solution.
                candidate_solution = self.neighbor(current_solution)
                candidate_cost, candidate_routes = self.problem.to_route(candidate_solution)
                delta = candidate_cost - current_cost
                
                # Accept the candidate if it's better, or with a probability if worse.
                if delta < 0 or random.random() < math.exp(-delta / self.T):
                    current_solution = candidate_solution
                    current_cost = candidate_cost
                    current_routes = candidate_routes
                    # Update best solution if improvement is found.
                    if candidate_cost < best_cost:
                        best_solution = candidate_solution
                        best_cost = candidate_cost
                        best_routes = candidate_routes
            
            # Cool down the temperature.
            self.T *= self.cooling_rate
        
        return best_solution, best_cost, best_routes
    
if __name__ == "__main__":
    # Assuming you have already created and loaded your Problem instance.
    problem = Problem()
    problem.load_data("data/25/C101_co_25.txt")  # Make sure your data file is correctly formatted.
    
    # Create an instance of the Simulated Annealing optimizer.
    sa = SimulatedAnnealing(problem, init_temperature=1000.0, cooling_rate=0.995, 
                            min_temperature=1e-3, iterations_per_temp=100)
    
    # Run the optimization.
    best_solution, best_cost, best_routes = sa.run()
    
    print("Distance: ", best_cost)
    print("Number of routes: ", len(best_routes))
    print("Longest route: ", max([len(route) for route in best_routes]))
    print("Shortest route: ", min([len(route) for route in best_routes]))