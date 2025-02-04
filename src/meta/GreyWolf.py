import random
import matplotlib.pyplot as plt
import numpy as np

from meta.solver import Solver

class GreyWolfOptimization(Solver):
    def __init__(self, objective_function, num_wolves, num_iterations, search_space):
        """
        Initializes the Grey Wolf Optimization instance.
        
        Parameters:
            objective_function (callable): The function to be minimized.
            num_wolves (int): The number of wolves (candidate solutions).
            num_iterations (int): The maximum number of iterations.
            search_space (list of tuples): The bounds for each dimension.
        """
        super().__init__(objective_function, num_iterations=num_iterations, search_space=search_space)
        self.num_wolves = num_wolves
        self.wolves = [self._initialize_wolf() for _ in range(num_wolves)]
    
    def _initialize_wolf(self):
        """
        Initializes a wolf's position randomly within the search space.
        """
        return [random.uniform(self.search_space[i][0], self.search_space[i][1]) 
                for i in range(len(self.search_space))]
    
    def optimize(self, verbose=True):
        """
        Executes the GWO algorithm and returns the best solution found and its fitness.
        """
        for iteration in range(self.num_iterations):
            # Evaluate fitness for each wolf
            fitnesses = [self.objective_function(wolf) for wolf in self.wolves]
            
            # Identify the top three wolves (alpha, beta, and delta)
            sorted_indices = sorted(range(self.num_wolves), key=lambda i: fitnesses[i])
            alpha = self.wolves[sorted_indices[0]]
            beta = self.wolves[sorted_indices[1]] if self.num_wolves > 1 else alpha
            delta = self.wolves[sorted_indices[2]] if self.num_wolves > 2 else beta
            
            # Update the overall best solution
            if fitnesses[sorted_indices[0]] < self.global_best_fitness:
                self.global_best_fitness = fitnesses[sorted_indices[0]]
                self.global_best_position = alpha.copy()
            
            # Coefficient 'a' decreases linearly from 2 to 0 over iterations
            a = 2 - iteration * (2 / self.num_iterations)
            
            new_wolves = []
            # Update the position of each wolf
            for wolf in self.wolves:
                new_position = []
                for j in range(len(self.search_space)):
                    # Update with respect to the alpha wolf
                    r1 = random.random()
                    r2 = random.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha[j] - wolf[j])
                    X1 = alpha[j] - A1 * D_alpha
                    
                    # Update with respect to the beta wolf
                    r1 = random.random()
                    r2 = random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta[j] - wolf[j])
                    X2 = beta[j] - A2 * D_beta
                    
                    # Update with respect to the delta wolf
                    r1 = random.random()
                    r2 = random.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta[j] - wolf[j])
                    X3 = delta[j] - A3 * D_delta
                    
                    # New position is the average of X1, X2, and X3
                    new_j = (X1 + X2 + X3) / 3
                    
                    # Ensure the new position is within the search space bounds
                    lower_bound = self.search_space[j][0]
                    upper_bound = self.search_space[j][1]
                    new_j = max(lower_bound, min(new_j, upper_bound))
                    
                    new_position.append(new_j)
                new_wolves.append(new_position)
            
            # Update the wolves' positions for the next iteration
            self.wolves = new_wolves
            
            self.fitness_history.append(self.global_best_fitness)
            if verbose:
                print(f"Iteration {iteration + 1}: Best Fitness = {self.global_best_fitness}")
        
        return self.global_best_position, self.global_best_fitness