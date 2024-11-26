# Implemet Simulated Annealing algorithm to solve the VRP with local search

import random
import math
from utils.route import Solution

class SimulatedAnnealingSolver:
    def __init__(self, initial_solution: Solution, initial_temp: float, cooling_rate: float, stopping_temp: float):
        self.current_solution = initial_solution
        self.best_solution = initial_solution
        self.current_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.stopping_temp = stopping_temp

    def objective_function(self, solution: Solution) -> float:
        # Define the objective function to evaluate the solution
        result = solution.total_distance()
        assert result <= 10000
        return result

    def local_search(self, solution: Solution) -> Solution:
        # Apply local search operations: decompose, swap, merge
        new_solution: Solution = solution.copy()
        # Check if new_solution is a copy of solution
        assert new_solution is not solution, "New solution is a copy of the original solution"
        operation = random.choice(['swap', 'merge'])
        # if operation == 'decompose':
        #     new_solution.random_decompose()
        if operation == 'swap':
            print("Swapping...")
            succss = new_solution.random_swap()
            if not succss:
                print("Swapping failed")
        elif operation == 'merge':
            print("Merging...")
            new_solution.random_merge()
        return new_solution

    def accept_solution(self, candidate_cost: float) -> bool:
        current_cost = self.objective_function(self.current_solution)
        if candidate_cost < current_cost:
            return True
        else:
            acceptance_prob = math.exp((current_cost - candidate_cost) / self.current_temp)
            return random.random() < acceptance_prob

    def anneal(self):
        history = {"iteration": [], "current_cost": [], "best_cost": []}

        iteration = 0
        print(f"Initial temperature: {self.current_temp:.2f}")
        print(f"Initial solution cost: {self.objective_function(self.current_solution):.2f}")
        
        while self.current_temp > self.stopping_temp:
            iteration += 1
            candidate_solution = self.local_search(self.current_solution)
            candidate_cost = self.objective_function(candidate_solution)
            if self.accept_solution(candidate_cost):
                self.current_solution = candidate_solution
                if candidate_cost < self.objective_function(self.best_solution):
                    self.best_solution = candidate_solution
                    print(f"Iteration {iteration}: New best solution found!")
                    print(f"Temperature: {self.current_temp:.2f}")
                    print(f"Cost: {self.objective_function(self.best_solution):.2f}")
                    print(f"Num graphs: {len(self.best_solution.graphs)}")
                    print("-" * 50)
            self.current_temp *= self.cooling_rate
            
            # Print progress every 100 iterations
            if iteration % 10 == 0:
                current_cost = self.objective_function(self.current_solution)
                best_cost = self.objective_function(self.best_solution)
                history["iteration"].append(iteration)
                history["current_cost"].append(current_cost)
                history["best_cost"].append(best_cost)
                print(f"Iteration {iteration}")
                print(f"Current temperature: {self.current_temp:.2f}")
                print(f"Current cost: {current_cost:.2f}")
                print(f"Best cost: {best_cost:.2f}")
                print("-" * 50)
        
        print("\nSimulated Annealing completed!")
        print(f"Final temperature: {self.current_temp:.2f}")
        print(f"Best solution cost: {self.objective_function(self.best_solution):.2f}")
        return self.best_solution, history