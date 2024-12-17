# Implemet Simulated Annealing algorithm to solve the VRP with local search

import random
from typing import List
from utils.Particle import PSOParticle
from utils.load_data import Location, OrderItem, Vehicle
# from utils.route import SASolution
import numpy as np

np.random.seed(42)
random.seed(42)


class PSOSolver:
    def __init__(self):
        self.n_particles = 10
        self.n_iterations = 50
        self.g_best = [] # Best position
        self.g_fitness = 0 # Best fitness
        self.particles: List[PSOParticle] = []
        self.locations: List[Location] = [] # [1, 2R]; R = number of orders
    
    def reset_swarm(self):
        self.g_best = []
        self.g_fitness = 0
        self.particles = []
        self.locations = []
        return
    
    def init_swarm(self, orders: List[OrderItem], vehicles: List[Vehicle]):
        self.reset_swarm()
        p_best = []
        p_fitness = 0
        for _ in range(self.n_particles):
            particle = PSOParticle(orders, vehicles)
            self.particles.append(particle)
            if particle.best_fitness < p_fitness or p_fitness == 0:
                p_best = particle.best_position
                p_fitness = particle.best_fitness
        self.g_best = p_best
        self.g_fitness = p_fitness
        for order in orders:
            self.locations.append(order.start_location)
            self.locations.append(order.end_location)
        return
    
    def fitness(self):
        """
        Update the fitness of each particle and update the global
        """
        for particle in self.particles:
            particle.decode()
            # particle.print_solution()
            particle.update_fitness()
            if particle.best_fitness < self.g_fitness or self.g_fitness == 0:
                self.g_best = particle.best_position
                self.g_fitness = particle.best_fitness

    def solve(self):
        print("Initial Best Fitness:", self.g_fitness)
        for i in range(self.n_iterations):
            print(f"Iteration: {i+1}/{self.n_iterations}")
            for particle in self.particles:
                particle.update_velocity(self.g_best)
                particle.update_position()
                particle.decode()
                particle.evaluate_position(particle.positions)
                if particle.best_fitness < self.g_fitness:
                    self.g_best = particle.best_position
                    self.g_fitness = particle.best_fitness
            print(f"Global Best Fitness: {self.g_fitness:.2f}")

    def print_solution(self):
        for particle in self.particles:
            particle.decode()
            particle.print_solution()
            print("-"*10)
        return
    
    def print_best_solution(self):
        best_particle = None
        best_fitness = 0
        for particle in self.particles:
            if particle.best_fitness < best_fitness or best_fitness == 0:
                best_particle = particle
                best_fitness = particle.best_fitness
        best_particle.decode()
        best_particle.print_solution()
        print("-"*10)
        return
    
    def visualize_solution(self):
        pass
        # plt.figure(figsize=(10, 8))
        # colors = plt.cm.get_cmap('tab20', len(self.vehicles))
        # for idx, order_set in enumerate(self.p_best_order_sets):
        #     if not order_set.orders:
        #         continue
        #     route = self.solutions[idx].route  # Assuming Route has a 'route' attribute with ordered locations
        #     x = [loc.x for loc in route]
        #     y = [loc.y for loc in route]
        #     plt.plot(x, y, marker='o', color=colors(idx), label=f'Vehicle {idx+1}')
        # plt.title('Routing Solution')
        # plt.xlabel('X Coordinate')
        # plt.ylabel('Y Coordinate')
        # plt.legend()
        # plt.show()