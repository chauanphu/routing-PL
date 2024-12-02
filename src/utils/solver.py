# Implemet Simulated Annealing algorithm to solve the VRP with local search

import random
import math
from typing import List
from utils.load_data import Location, OrderItem
from utils.route import SASolution
import numpy as np
import math 

np.random.seed(42)
random.seed(42)

class SimulatedAnnealingSolver:
    def __init__(self, initial_solution: SASolution, initial_temp: float, cooling_rate: float, stopping_temp: float):
        self.current_solution = initial_solution
        self.best_solution = initial_solution
        self.current_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.stopping_temp = stopping_temp

    def objective_function(self, solution: SASolution) -> float:
        # Define the objective function to evaluate the solution
        result = solution.total_distance()
        assert result <= 10000
        return result

    def local_search(self, solution: SASolution) -> SASolution:
        # Apply local search operations: decompose, swap, merge
        new_solution: SASolution = solution.copy()
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
    
class PSOParticle:
    def __init__(self, dim: int, orders: List[OrderItem]):
        self.positions = np.random.rand(dim)
        self.fitness = 0
        self.velocity = np.random.rand(dim)
        self.best_position = []
        self.best_fitness = 0
        self.weight = 0 # [0, 1]
        self.inertia = 0.72984 # [0, 1]
        self.c1 = 2.05 # [0, 2]
        self.c2 = 2.05 # [0, 2]

        self.orders = orders

    def update_position(self):
        self.positions += self.velocity

    def update_velocity(self):
        self.velocity = self.inertia * self.velocity + self.c1 * np.random.rand() * (self.best_position - self.positions) + self.c2 * np.random.rand() * (self.best_position - self.positions)

    def update_fitness(self):
        pass

    def __repr__(self) -> str:
        return f"Particle: {self.positions} Fitness: {self.fitness:.2f}"

class PSOSolver:
    def __init__(self):
        self.n_particles = 10
        self.n_iterations = 100
        self.p_best = []
        self.p_fitness = 0
        self.g_best = []
        self.g_fitness = 0
        self.particles: List[PSOParticle] = []
        self.locations: List[Location] = [] # [1, 2R]; R = number of orders

    def init_swarm(self, orders: List[OrderItem], n_vehicle: int):
        num_orders = len(orders)
        for _ in range(self.n_particles):
            particle = PSOParticle(2 * num_orders)
            self.particles.append(particle)
        for order in orders:
            self.locations.append(order.start_location)
            self.locations.append(order.end_location)
        return
    
    def solve(self):
        for i in range(self.n_iterations):
            print(f"Iteration: {i+1}/{self.n_iterations}")
            for particle in self.particles:
                particle.update_velocity()
                particle.update_position()
                particle.update_fitness()
                if particle.best_fitness > self.p_fitness:
                    self.p_best = particle.best_position
                    self.p_fitness = particle.best_fitness
            if self.p_fitness > self.g_fitness:
                self.g_best = self.p_best
                self.g_fitness = self.p_fitness
        pass

    def decode(self):
        pass

    def stop_condition(self) -> bool:
        pass