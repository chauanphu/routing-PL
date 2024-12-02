# Implemet Simulated Annealing algorithm to solve the VRP with local search

import random
import math
from typing import List
from utils.load_data import Location, OrderItem, Vehicle
# from utils.route import SASolution
import numpy as np
import math
import networkx as nx
from utils.route import OrderSet 

np.random.seed(42)
random.seed(42)

class PSOParticle:
    def __init__(self, orders: int, vehicles: int):
        num_order = len(orders)
        num_vehicle = len(vehicles)

        self.positions = np.random.randint(0, num_vehicle, num_order) # Dim: [1, num_order], Value in [0, num_vehicle]
        self.velocity = np.random.rand(num_order)
        self.fitness = 0
        self.best_position = []
        self.best_fitness = 0
        self.weight = 0 # [0, 1]
        self.inertia = 0.72984 # [0, 1]
        self.c1 = 2.05 # [0, 2]
        self.c2 = 2.05 # [0, 2]
        
        self.orders: List[OrderItem] = orders
        self.order_sets: List[OrderSet] = []
        self.vehicles: List[Vehicle] = vehicles

    def assign(self):
        self.order_sets: List[OrderSet] = [OrderSet(v.capacity) for v in self.vehicles]
        assignment = np.round(self.positions)
        for i in range(len(assignment)):
            vehicle_id = int(assignment[i])
            order = self.orders[i]
            self.order_sets[vehicle_id].add_order(order)

    def print_solution(self):
        for order_set in self.order_sets:
            if not order_set.orders:
                continue
            print("Orders")
            for order in order_set.orders.values():
                print("-", order)
            print("Route")
            if nx.is_directed_acyclic_graph(order_set):
                route = order_set.weighted_topological_sort(weight="due_time")
                print(route)
            else:
                print("Not a DAG")
            print("-"*10)

    def update_position(self):
        self.positions += self.velocity

    def update_velocity(self):
        self.velocity = self.inertia * self.velocity + self.c1 * np.random.rand() * (self.best_position - self.positions) + self.c2 * np.random.rand() * (self.best_position - self.positions)

    def update_fitness(self):
        total_distance = 0
        for order_set in self.order_sets:
            if not order_set.orders:
                continue
            for order in order_set.orders.values():
                print(order)
            if not nx.is_directed_acyclic_graph(order_set):
                self.fitness = 10000 # A very large number
                return self.fitness
            
            _, total_distance = order_set.weighted_topological_sort(weight="due_time")
            self.fitness += total_distance
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.positions

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

    def init_swarm(self, orders: List[OrderItem], vehicles: List[Vehicle]):
        for _ in range(self.n_particles):
            particle = PSOParticle(orders, vehicles)
            self.particles.append(particle)

        for order in orders:
            self.locations.append(order.start_location)
            self.locations.append(order.end_location)
        return
    
    def fitness(self):
        for particle in self.particles:
            particle.assign()
            particle.print_solution()

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

    def decode(self):
        pass

    def stop_condition(self) -> bool:
        pass