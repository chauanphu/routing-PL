import networkx as nx
from utils.route import OrderSet, Route 
from typing import List, Tuple
from utils.load_data import OrderItem, Vehicle
import numpy as np
from utils.config import COGNITIVE_WEIGHT, SOCIAL_WEIGHT, INFEASIBILITY_PENALTY, INERTIA, ALLOW_EARLY

class PSOParticle:
    def __init__(self, orders: List[OrderItem], vehicles: List[Vehicle]):
        num_order = len(orders)
        num_vehicle = len(vehicles) - 1
        self.orders: List[OrderItem] = orders
        self.order_sets: List[OrderSet] = []
        self.vehicles: List[Vehicle] = vehicles
        self.solutions: List[Route] = []

        self.positions = np.random.uniform(0, num_vehicle, num_order * 2) # Dim: [1, 2 * num_order], pairs of (assigned vehicle, priority)
        
        self.velocity = np.random.rand(num_order * 2) # Dim: [1, 2 * num_order]
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

    def decode(self) -> Tuple[List[OrderSet], List[bool]]:
        """
        Decoding function, assign orders to vehicles based on the position

        ## Mechanism

        1. Create a list of OrderSet for each vehicle
        2. Assign orders to vehicles based on the position
        3. Update the depot of each OrderSet
        """
        self.order_sets: List[OrderSet] = [OrderSet(capacity=v.capacity, depot=v.start) for v in self.vehicles]
        # The first half of the position is the vehicle assignment, the second half is the priority
        assignment = self.positions[:len(self.orders)]
        priority = self.positions[len(self.orders):]
        assignment = np.round(assignment)
        i = 0
        ## Assign orders to vehicles
        for _assign, _priority, order in zip(assignment, priority, self.orders):
            vehicle_id = int(_assign)
            order_set = self.order_sets[vehicle_id]
            previous_order = len(order_set.orders)
            order_set.add_order(order, priority=float(_priority))
            assert len(order_set.orders) == previous_order + 1, "Order not added"
            i += 1
        self.order_sets = [o for o in self.order_sets if not o.isEmpty()]
        served_orders = sum([len(o.orders) for o in self.order_sets])
        assert served_orders == len(self.orders), f"Served orders not equal to total orders: {served_orders} != {len(self.orders)}"

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
                route = order_set.weighted_topological_sort(weight="due_time", allow_early=ALLOW_EARLY)
                print(route)
            else:
                print("Not a DAG")
            print("-"*10)

    def update_position(self):
        self.positions += self.velocity
        self.positions = np.clip(self.positions, 0, len(self.vehicles) - 1, out=self.positions)
        # assert np.equal(previous_pos, self.positions).all() == False, "Position not updated"
       
    def update_velocity(self, global_best: np.ndarray):
        cognitive_component = self.c1 * np.random.rand() * (self.p_best - self.positions)
        social_component = self.c2 * np.random.rand() * (global_best - self.positions)
        self.velocity = self.inertia * self.velocity + cognitive_component + social_component

    def update_fitness(self):
        total_distance = 0
        solutions = []
        fitness = 0
        for order_set in self.order_sets:
            if not nx.is_directed_acyclic_graph(order_set):
                fitness += INFEASIBILITY_PENALTY # A very large number
                solutions.append(None)
                continue
            try:
                route, total_distance = order_set.weighted_topological_sort(weight="due_time", allow_early=ALLOW_EARLY)
            except nx.NetworkXUnfeasible as e:
                # print(e)
                fitness += INFEASIBILITY_PENALTY
                solutions.append(None)
                continue
            
            fitness += float(total_distance)
            solutions.append(route)
            
        if self.p_fitness:
            if fitness > self.p_fitness:
                return
        self.p_fitness = fitness
        self.p_best = np.copy(self.positions)
        self.p_solution = solutions
        
    def __repr__(self) -> str:
        return f"Particle: {self.positions} Fitness: {self.fitness:.2f}"
