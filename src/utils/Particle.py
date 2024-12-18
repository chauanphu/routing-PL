import networkx as nx
from utils.route import OrderSet, Route 
from typing import List
from utils.load_data import OrderItem, Vehicle
import numpy as np
from utils.config import COGNITIVE_WEIGHT, SOCIAL_WEIGHT, INFEASIBILITY_PENALTY, INERTIA

class PSOParticle:
    def __init__(self, orders: List[OrderItem], vehicles: List[Vehicle]):
        num_order = len(orders)
        num_vehicle = len(vehicles) - 1
        self.orders: List[OrderItem] = orders
        self.order_sets: List[OrderSet] = []
        self.vehicles: List[Vehicle] = vehicles
        self.solutions: List[Route] = []

        self.positions = np.random.uniform(0, num_vehicle, num_order) # Dim: [1, num_order], Value in [0, num_vehicle)
        
        self.velocity = np.random.rand(num_order)
        self.p_best = None
        self.p_fitness = None
        # Parameters
        self.inertia = INERTIA
        self.c1 = COGNITIVE_WEIGHT
        self.c2 = SOCIAL_WEIGHT

    def setup(self):
        """
        Setup the particle with random positions and velocities
        """
        self.decode()
        self.update_fitness()

    def decode(self):
        """
        Decoding function, assign orders to vehicles based on the position

        ## Mechanism

        1. Create a list of OrderSet for each vehicle
        2. Assign orders to vehicles based on the position
        3. Update the depot of each OrderSet
        """
        self.order_sets: List[OrderSet] = [OrderSet(capacity=v.capacity, depot=v.start) for v in self.vehicles]
        assignment = np.round(self.positions)
        ## Assign orders to vehicles
        for i in range(len(assignment)):
            vehicle_id = int(assignment[i]) - 1
            order = self.orders[i]
            self.order_sets[vehicle_id].add_order(order)
        self.order_sets = [o for o in self.order_sets if o.orders]
            
    def print_solution(self):
        print("Solution")
        print("Used Vehicles:", len([o for o in self.order_sets if o.orders]))
        print("-"*20)
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
        previous_pos = np.copy(self.positions)
        self.positions += self.velocity
        self.positions = np.clip(self.positions, 0, len(self.vehicles) - 1, out=self.positions)
        assert np.equal(previous_pos, self.positions).all() == False, "Position not updated"
       
    def update_velocity(self, global_best: np.ndarray):
        cognitive_component = self.c1 * np.random.rand() * (self.p_best - self.positions)
        social_component = self.c2 * np.random.rand() * (global_best - self.positions)
        self.velocity = self.inertia * self.velocity + cognitive_component + social_component

    def update_fitness(self):
        total_distance = 0
        self.solutions = []
        fitness = 0
        for order_set in self.order_sets:
            if order_set.isEmpty():
                continue
            if not nx.is_directed_acyclic_graph(order_set):
                fitness = INFEASIBILITY_PENALTY # A very large number
                if self.p_fitness is None:
                    self.p_fitness = INFEASIBILITY_PENALTY
                    self.p_best = np.copy(self.positions)
                return
            try:
                route, total_distance = order_set.weighted_topological_sort(weight="due_time")
            except nx.NetworkXUnfeasible as e:
                # print(e)
                fitness = INFEASIBILITY_PENALTY
                if self.p_fitness is None:
                    self.p_fitness = INFEASIBILITY_PENALTY
                    self.p_best = np.copy(self.positions)
                return
            
            fitness += total_distance
            self.solutions.append(route)
        if self.p_fitness:
            if fitness > self.p_fitness:
                return
        self.p_fitness = fitness
        self.p_best = np.copy(self.positions)

    def evaluate_position(self, position: np.ndarray):
        """
        Evaluate the fitness of the swarms. Decode into route then calculate the total distance
        """
        order_sets: List[OrderSet] = [OrderSet(v.capacity) for v in self.vehicles]
        assignment = np.round(position)
        fitness = 0
        for i in range(len(assignment)):
            vehicle_id = int(assignment[i]) - 1
            vehicle = self.vehicles[vehicle_id]
            order = self.orders[i]
            order_sets[vehicle_id].add_order(order)
            order_sets[vehicle_id].depot = vehicle.start
        order_sets = [o for o in order_sets if o.orders]

        for order_set in order_sets:
            if order_set.isEmpty():
                continue
            
            if not nx.is_directed_acyclic_graph(order_set):
                return INFEASIBILITY_PENALTY
            try:
                _, total_distance = order_set.weighted_topological_sort(weight="due_time")
            except nx.NetworkXUnfeasible as e:
                print(e)
                return INFEASIBILITY_PENALTY
            fitness += total_distance
        return fitness
        

    def __repr__(self) -> str:
        return f"Particle: {self.positions} Fitness: {self.fitness:.2f}"
