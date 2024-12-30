# Implemet Simulated Annealing algorithm to solve the VRP with local search

import random
from typing import List
from meta.genetics.GA import GA
from meta.PSO.Particle import PSOParticle
from utils.load_data import Location, OrderItem, Vehicle
# from utils.route import SASolution
import numpy as np
from utils.config import GA_ENABLED
np.random.seed(42)
random.seed(42)


class PSOSolver:
    def __init__(self, n_particles=1000, n_iterations=50, orders: List[OrderItem] = None, vehicles: List[Vehicle] = None):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.g_best = [] # Best position
        self.g_fitness = 0 # Best fitness
        self.particles: List[PSOParticle] = []
        self.locations: List[Location] = [] # [1, 2R]; R = number of orders
        self.particle_history = [] # Add this line to store particle histories
        self.final_solution = {}
        self.orders = orders
        self.vehicles = vehicles

    def reset_swarm(self):
        self.g_best = []
        self.g_fitness = 0
        self.particles = []
        self.locations = []
        self.particle_history = [] # Add this line
        return
    
    def init_swarm(self):
        self.reset_swarm()
        p_best = []
        p_fitness = 0
        for _ in range(self.n_particles):
            particle = PSOParticle(self.orders, self.vehicles)
            particle.setup()
            self.particles.append(particle)
            if particle.p_fitness < p_fitness or p_fitness == 0:
                p_best = particle.p_best
                p_fitness = particle.p_fitness
        self.g_best = p_best
        self.g_fitness = p_fitness
        for order in self.orders:
            self.locations.append(order.start_location)
            self.locations.append(order.end_location)
        return

    def solve(self):
        print("Initial Best Fitness:", self.g_fitness)
        history = {
            'fitness': [],
            'particle_fitness': [[] for _ in range(self.n_particles)]  # Track each particle's fitness
        }
        
        for i in range(self.n_iterations):
            print(f"Iteration: {i+1}/{self.n_iterations}", end='\t')

            for idx, particle in enumerate(self.particles):
                particle.update_velocity(self.g_best)
                particle.update_position()
                particle.decode()
                particle.update_fitness()
            # Evolution of each individual
            if GA_ENABLED:
                ga = GA(i, self.particles, num_vehicle=len(self.vehicles)-1, num_orders=len(self.orders))
                self.particles = ga.evolve(self.orders, self.vehicles)
            # Evaluate the best fitness
            for idx, particle in enumerate(self.particles):
                if particle.p_fitness < self.g_fitness:
                    if None in particle.p_solution:
                        print("Invalid solution")
                    self.g_best = particle.p_best
                    self.g_fitness = particle.p_fitness
                    self.final_solution = {
                        "order_set": particle.order_sets,
                        "routes": particle.p_solution
                    }
                # Store particle's personal best fitness
                history['particle_fitness'][idx].append(particle.p_fitness)

            history['fitness'].append(self.g_fitness)
            print(f"Global Best Fitness: {self.g_fitness:.2f}")

        return history
    
    def print_solution(self):
        for particle in self.particles:
            particle.decode()
            particle.print_solution()
            print("-"*10)
        return
    
    def print_best_solution(self, file_name='output/pso.best_solution.txt'):
        with open(file_name, 'w') as f:
            if not self.final_solution:
                f.write("No solution found\n")
                return
            # Print configuration
            f.write(f"Number of particles: {self.n_particles}\n")
            f.write(f"Number of iterations: {self.n_iterations}\n")
            f.write("-"*20 + "\n")
            # Print the metadata
            f.write(f"Best Fitness: {self.g_fitness}\n")
            f.write(f"Number of vehicles used: {len(self.final_solution['routes'])}\n")
            f.write(f"Number of orders: {sum([len(o.orders) for o in self.final_solution['order_set']])}\n")
            f.write("Best Solution\n")
            f.write("-"*20 + "\n")
            for idx, (route, order_set) in enumerate(zip(self.final_solution['routes'], self.final_solution['order_set'])):
                f.write(f"Route {idx+1}\n")
                f.write("-"*10 + "\n")
                for order in order_set.orders.values():
                    f.write(f"{order}\n")
                f.write("Route\n")
                f.write(f"{route}\n")
                f.write("-"*10 + "\n")
                f.write("\n")