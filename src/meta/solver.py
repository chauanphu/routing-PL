import math
import time
from typing import Any, List, Tuple, Union
import numpy as np
from pydantic import BaseModel
from matplotlib import pyplot as plt
import random

class Node(BaseModel):
    """
    VRPPL Node class: customers, depots, and lockers

    Storing the node data including the node id, x and y coordinates
    """
    node_id: int
    x: float
    y: float
    early: float
    late: float
    service: float

class Customer(Node):
    """
    VRPPL Customer class

    Storing the customer data including the customer id, x and y coordinates, early time, late time, service time, and demand
    """
    demand: float
    customer_type: int
    assigned_locker: Node | None = None
    locker_delivery: bool | None = None

class Vehicle(BaseModel):
    """
    VRPPL Vehicle class

    Storing the vehicle data including the vehicle id, capacity, and fixed cost
    """
    v_id: int
    capacity: float
    load: float
    current_time: float
    visited: list[int] = []

    # Ensuring that the vehicle load is not greater than the vehicle capacity
    def __init__(self, **data):
        super().__init__(**data)
        if self.load > self.capacity:
            raise ValueError("Vehicle load cannot be greater than vehicle capacity")
        
    def add_demand(self, demand):
        self.load += demand
        if self.load > self.capacity:
            raise ValueError("Vehicle load cannot be greater than vehicle capacity")

class Problem:
    """
    VRPPL Problem class

    Storing the problem data including nodes, vehicles, and demands
    """

    customers: list[Customer]
    depot: Node
    lockers: list[Node]
    num_vehicles: int
    num_lockers: int
    num_customers: int
    vehicle_capacity: int

    def load_data(self, path: str):
        """
        Load the data from the input file

        Parameters:
        path (str): the path to the input file
        
        ## Description of file content:       
        - <Number of customers>\t<Number of lockers> 
        - <Number of vehicles>\t<Vehicle capacity>
        - num_customers X <Customer demands>
        - <Depot'x coordinate>\t<Depot's y coordinate>\t<Depot's early time>\t<Depot's late time>\t<Depot's service time>\t<Customer type = 0>
        - num_customers X <Customer'x coordinate>\t<Customer's y coordinate>\t<Customer's early time>\t<Customer's late time>\t<Customer's service time>\t<Customer type = 1, 2, 3>
        - num_lockers X <Locker'x coordinate>\t<Locker's y coordinate>\t<Locker's early time>\t<Locker's late time>\t<Locker's service time>\t<Locker type = 4>
        - num_customers X <Binary encoded locker assignment for each customer>. Ex: 0 1 => customer 1 is assigned to locker 2
        """
        with open(path, 'r') as f:
            # Read the first line to get the number of customers and lockers
            self.num_customers, self.num_lockers = map(int, f.readline().split())
            self.num_vehicles, self.vehicle_capacity = map(int, f.readline().split())
            # Read customer demands
            demands = []
            for _ in range(self.num_customers):
                demands.append(int(f.readline()))
            # Read depot data
            x, y, early, late, service, _ = map(float, f.readline().split())
            self.depot = Node(node_id=0, x=x, y=y, early=early, late=late, service=service)
            
            # Read customer data
            self.customers = []
            for i in range(1, self.num_customers + 1):
                x, y, early, late, service, customer_type = map(float, f.readline().split())
                self.customers.append(Customer(node_id=i, x=x, y=y, early=early, late=late, service=service, demand=demands[i-1], customer_type=customer_type))
            # Read locker data
            self.lockers = []
            for i in range(1, self.num_lockers + 1):
                x, y, early, late, service, _ = map(float, f.readline().split())
                self.lockers.append(Node(node_id=self.num_customers + i, x=x, y=y, early=early, late=late, service=service))
            # Read locker assignment data
            for i in range(self.num_customers):
                for j, assignment in enumerate(map(int, f.readline().split())):
                    if assignment:
                        self.customers[i].assigned_locker = self.lockers[j]
                        break

    @staticmethod
    def euclidean_distance(n1: Node, n2: Node) -> float:
        """Helper function to calculate Euclidean distance between two nodes."""
        return math.hypot(n1.x - n2.x, n1.y - n2.y)

    def select_nearest_locker(
        self, last_node: 'Node'
    ) -> Union['Node', None]:
        """
        From the available parcel lockers, select the one that is closest to the current
        location (last_node) and can be feasibly visited (i.e. its time window is met)
        from the current_vehicle's state.
        """
        best_locker = None
        best_distance = float('inf')
        for locker in self.lockers:
            travel_time = self.euclidean_distance(last_node, locker)
            if travel_time < best_distance:
                best_distance = travel_time
                best_locker = locker
        return best_locker

    def initialize_solution(self, p: float) -> Tuple[List[Any], float]:
        # Copy of customers to be assigned.
        sorted_customers = sorted(self.customers, key=lambda cust: cust.node_id)
        # For each customer, assign a delivery node based on type.
        # For type 1, assign the customer itself.
        # For type 2 and type 3, assign the selected (nearest) parcel locker station.
        assignment_list = []
        for customer in sorted_customers:
            chosen_node, chosen_node = self._locker_assignment(customer, explore=False, p=p)
            assignment_list.append((customer, chosen_node))
        
        # --- Step 2: Nearest-Neighbor Ordering ---
        # Initialize the two-row representation.
        first_row = [0]             # Row 0: depot marker at the beginning.
        second_row = [self.depot]     # Row 1: actual visited nodes, starting with depot.
        current_node = self.depot

        # While there remain unassigned customers, select the one whose assigned node
        # is closest (in Euclidean distance) to the current node.
        while assignment_list:
            candidate_tuple = min(
                assignment_list,
                key=lambda tup: self.euclidean_distance(current_node, tup[1])
            )
            customer, chosen_node = candidate_tuple
            first_row.append(customer.node_id)
            second_row.append(chosen_node)
            current_node = chosen_node
            assignment_list.remove(candidate_tuple)
        
        # Terminate the route by appending the depot at the end.
        first_row.append(0)
        second_row.append(self.depot)

        solution = [first_row, second_row]
        
        return solution
    
    def initialize_position(self, p: float) -> list[float]:
        """
        Generate an initial continuous position vector for customers in [0,1) such that
        customers that are near each other (based on their chosen delivery node, taking into
        account locker assignment) get positions with similar values.
        
        The procedure is as follows:
        1. For each customer, decide its locker delivery preference via _locker_assignment.
        2. Starting at the depot, build a nearest-neighbor ordering using the chosen delivery node.
        3. Assign positions in increasing order along [0,1), with a small random perturbation.
        
        Returns:
            positions (list[float]): A list of continuous values (one per customer) in [0,1).
                                    When sorted, the customer order reflects the nearest-neighbor ordering.
        """
        # Step 1: Create a list of (customer, chosen_node) tuples using the locker assignment.
        assignment_list = []
        for customer in self.customers:
            chosen_node, _ = self._locker_assignment(customer, explore=False, p=p)
            assignment_list.append((customer, chosen_node))
        
        # Step 2: Build a nearest-neighbor ordering starting from the depot.
        ordered_customers = []
        current_node = self.depot
        while assignment_list:
            # Find the tuple (customer, chosen_node) whose chosen_node is nearest to current_node.
            best_tuple = min(assignment_list, key=lambda tup: self.euclidean_distance(current_node, tup[1]))
            ordered_customers.append(best_tuple[0])
            current_node = best_tuple[1]
            assignment_list.remove(best_tuple)
        
        # Step 3: Assign continuous positions based on the ordering.
        n = len(self.customers)
        positions = [0.0] * n  # Placeholder: one position per customer.
        
        # We assign positions in increasing order. Adding a small noise helps keep values unique.
        for order, customer in enumerate(ordered_customers):
            # Evenly space the positions in [0,1). For example, position base = order / n.
            base = order / n
            # Add a small noise in the range [-0.5/n, 0.5/n].
            noise = random.uniform(-0.5 / n, 0.5 / n)
            pos = base + noise
            # Ensure that the final value remains in [0,1)
            pos = max(0.0, min(1.0 - 1e-8, pos))
            # Place the position in the output list.
            # Here we assume that self.customers is stored in a fixed order and that the index of a customer
            # in self.customers corresponds to its dimension in the positions vector.
            index = self.customers.index(customer)
            positions[index] = pos
    
        return positions

    def get_search_space(self) -> list[tuple[float, float]]:
        """
        Get the search space for the problem.

        Returns:
            search_space (list[tuple[float, float]]): The search space for the problem.
        """
        return [(0.0, 1.0) for _ in range(self.num_customers)]

    def position2route(self, positions: list[float]) -> tuple[float, list[list[Node]]]:
        """
        Decode the positions (continuous values) to a permutation of customer visits,
        perform locker assignment similar to permu2route, split the permutation into routes 
        for vehicles (with depot at the beginning and end of each route), and compute the total 
        travel distance. If a route is infeasible (due to time windows, capacity, or duplicate locker
        visits), a penalty (float('inf')) is returned.
        
        Parameters:
            positions (list[float]): A list of continuous values (one per customer) in [0,1).
        
        Returns:
            A tuple (total_distance, routes) where:
            - total_distance (float): The sum of travel distances of all routes (or float('inf') if infeasible).
            - routes (list[list[Node]]): A list of routes (each route is a list of Node objects).
        """
        # --- Step 1: Decode Positions to an Ordered List of Customers ---
        # Sort customer indices based on the continuous values.
        sorted_indices = sorted(range(len(positions)), key=lambda i: positions[i])
        ordered_customers = [self.customers[i] for i in sorted_indices]
        
        # --- Step 2: Initialize Route and Vehicle State ---
        routes = []
        total_distance = 0.0

        current_route = [self.depot]  # Each route starts with the depot.
        vehicle_index = 0
        current_vehicle = Vehicle(
            v_id=vehicle_index,
            capacity=self.vehicle_capacity,
            load=0.0,
            current_time=0.0
        )
        last_node = self.depot
        route_distance = 0.0
        visited: set[int] = set()  # To check duplicate locker visits.
        
        # --- Step 3: Process Each Customer in the Order ---
        for customer in ordered_customers:
            # Locker Assignment: decide whether to deliver to the customer or its assigned locker.
            chosen_node, _ = self._locker_assignment(customer, explore=False)
            
            # Compute travel time from the last node to the chosen node.
            travel_time = self.euclidean_distance(last_node, chosen_node)
            arrival_time = current_vehicle.current_time + travel_time
            if arrival_time < chosen_node.early:
                arrival_time = chosen_node.early

            # Check if this customer (via chosen node) can be feasibly served:
            # - Time window is met.
            # - Vehicle has enough capacity.
            # - A locker is not visited twice in a non-consecutive manner.
            duplicate_lockers = (
                (chosen_node.node_id in visited) and 
                (chosen_node.node_id != self.depot.node_id) and 
                (chosen_node.node_id != last_node.node_id)
            )
            if (arrival_time > chosen_node.late) or (current_vehicle.load + customer.demand > current_vehicle.capacity) or duplicate_lockers:
                # End current route: return to depot.
                return_to_depot = self.euclidean_distance(last_node, self.depot)
                route_distance += return_to_depot
                current_route.append(self.depot)
                total_distance += route_distance
                routes.append(current_route)

                # Switch to the next available vehicle.
                vehicle_index += 1
                if vehicle_index >= self.num_vehicles:
                    return float('inf'), []  # Infeasible: ran out of vehicles.
                
                current_vehicle = Vehicle(
                    v_id=vehicle_index,
                    capacity=self.vehicle_capacity,
                    load=0.0,
                    current_time=0.0
                )
                # Reset the route details.
                current_route = [self.depot]
                last_node = self.depot
                route_distance = 0.0
                visited.clear()

                # Recalculate travel details from the depot to the chosen node.
                travel_time = self.euclidean_distance(last_node, chosen_node)
                arrival_time = current_vehicle.current_time + travel_time
                if arrival_time < chosen_node.early:
                    arrival_time = chosen_node.early
                if (arrival_time > chosen_node.late) or (current_vehicle.load + customer.demand > current_vehicle.capacity):
                    return float('inf'), []  # Even after starting a new route, the node is infeasible.
            
            # --- Accept the Customer (via chosen_node) ---
            route_distance += travel_time
            current_route.append(chosen_node)
            current_vehicle.current_time = arrival_time + chosen_node.service
            current_vehicle.load += customer.demand
            visited.add(chosen_node.node_id)
            last_node = chosen_node

        # --- Step 4: Finalize the Last Route ---
        return_to_depot = self.euclidean_distance(last_node, self.depot)
        route_distance += return_to_depot
        current_route.append(self.depot)
        total_distance += route_distance
        routes.append(current_route)

        return total_distance, routes
    
    def route_evaluate(self, routes: List[List[Node]]) -> float:
        """
        Evaluate a given set of routes (each route is a list of Node objects)
        by computing the total travel distance while checking feasibility in terms of
        time windows and vehicle capacity.
        
        Feasibility checks:
          - The arrival time at each node must lie within its [early, late] time window.
          - The cumulative customer demand along each route must not exceed the vehicle capacity.
        
        Each route is assumed to start and end at the depot. If not, the depot is added automatically.
        
        Parameters:
            routes (List[List[Node]]): A list of routes, where each route is a list of Node objects.
            
        Returns:
            total_distance (float): The total travel distance for all routes. If any route is infeasible,
                                      returns float('inf') as a penalty.
        """
        total_distance = 0.0

        # Evaluate each route independently.
        for route in routes:
            # Ensure the route starts and ends at the depot.
            if not route or route[0].node_id != self.depot.node_id:
                route = [self.depot] + route
            if route[-1].node_id != self.depot.node_id:
                route = route + [self.depot]

            # Initialize route variables.
            route_distance = 0.0
            current_time = 0.0  # Vehicle starts at time 0.0
            load = 0.0         # Vehicle load starts at 0.0

            # Process each leg of the route.
            for i in range(1, len(route)):
                prev_node = route[i - 1]
                current_node = route[i]

                # Calculate travel time/distance.
                travel_time = self.euclidean_distance(prev_node, current_node)
                route_distance += travel_time

                # Calculate arrival time at the current node.
                arrival_time = current_time + travel_time
                # Wait until the node's early time if arriving too early.
                if arrival_time < current_node.early:
                    arrival_time = current_node.early
                # If arrival is later than the node's late time, route is infeasible.
                if arrival_time > current_node.late:
                    return float('inf')

                # Update current time to include service time.
                current_time = arrival_time + current_node.service

                # If the current node is a customer, update the load.
                # (We assume that only Customer nodes have the attribute "demand".)
                if hasattr(current_node, "demand"):
                    load += current_node.demand
                    if load > self.vehicle_capacity:
                        return float('inf')
            
            total_distance += route_distance

        return total_distance
    
    def _locker_assignment(self, customer: Customer, explore=True, p=0.5) -> tuple[Node, Customer]:
        """
        Assign a customer to a parcel locker station based on the customer's type.
        For type 1 customers, the assigned node is the customer itself.
        For type 2 and type 3 customers, the assigned node is the nearest parcel locker station.
        
        Parameters:
            customer (Customer): The customer object to assign to a parcel locker.
        
        Returns:
            assigned_node (Node): The assigned parcel locker station.
        """
        if customer.customer_type == 1:
            customer.locker_delivery = False
            chosen_node = customer
        elif customer.customer_type == 2:
            customer.assigned_locker
            customer.locker_delivery = True
            chosen_node = customer.assigned_locker
        elif customer.locker_delivery is None:
            if customer.customer_type == 3 or explore:
                if random.random() < p:
                    chosen_node = customer.assigned_locker
                    customer.locker_delivery = True
                else:
                    chosen_node = customer
                    customer.locker_delivery = False
            else:
                chosen_node = customer
        else:
            chosen_node = customer
        return chosen_node, customer

    def permu2route(self, permutation: list[Node | Customer], explore=False) -> tuple[float, list[list[Node]]]:
        routes = []
        total_distance = 0.0

        # Initialize the current route with the depot.
        current_route = [self.depot]
        # Initialize vehicle state.
        vehicle_index = 0
        current_vehicle = Vehicle(v_id=vehicle_index, capacity=self.vehicle_capacity, load=0.0, current_time=0.0)
        last_node = self.depot
        route_distance = 0.0
        visited: set[int] = set()
        # Create a lookup dictionary for customers based on their node_id.
        customer_dict = {cust.node_id: cust for cust in self.customers}

        # Process each customer ID from the first row (ignoring the depot markers).
        for cid in permutation[1:-1]:
            customer = customer_dict.get(cid)
            if customer is None:
                # Skip if the customer is not found.
                continue

            # --- Locker Assignment ---
            chosen_node, customer = self._locker_assignment(customer, explore)

            # --- Feasibility Check ---
            travel_time = self.euclidean_distance(last_node, chosen_node)
            arrival_time = current_vehicle.current_time + travel_time
            if arrival_time < chosen_node.early:
                arrival_time = chosen_node.early
            # If a locker station is visited more than once (not consecutive), skip it.
            duplicate_lockers = (chosen_node.node_id in visited) and (chosen_node.node_id != self.depot.node_id) and (chosen_node.node_id != last_node.node_id)
            # Check time window and capacity feasibility.
            if (arrival_time > chosen_node.late) or (current_vehicle.load + customer.demand > current_vehicle.capacity) or duplicate_lockers:
                # Terminate the current route: return to depot.
                return_to_depot = self.euclidean_distance(last_node, self.depot)
                route_distance += return_to_depot
                current_route.append(self.depot)
                total_distance += route_distance
                routes.append(current_route)

                # Switch to a new vehicle.
                vehicle_index += 1
                if vehicle_index >= self.num_vehicles:
                    return float('inf'), []
                current_vehicle = Vehicle(v_id=vehicle_index, capacity=self.vehicle_capacity, load=0.0, current_time=0.0)
                # Reset route details.
                current_route = [self.depot]
                last_node = self.depot
                route_distance = 0.0
                visited.clear()

                # Recalculate travel details from depot to the chosen node.
                travel_time = self.euclidean_distance(last_node, chosen_node)
                arrival_time = current_vehicle.current_time + travel_time
                if arrival_time < chosen_node.early:
                    arrival_time = chosen_node.early
                if (arrival_time > chosen_node.late) or (current_vehicle.load + customer.demand > current_vehicle.capacity):
                    return float('inf'), []

            # --- Accept the Customer ---
            route_distance += travel_time
            current_route.append(chosen_node)
            # Update vehicle state: add service time and customer demand.
            current_vehicle.current_time = arrival_time + chosen_node.service
            current_vehicle.load += customer.demand
            visited.add(chosen_node.node_id)
            last_node = chosen_node

        # Finalize the last route: return to depot.
        return_to_depot = self.euclidean_distance(last_node, self.depot)
        route_distance += return_to_depot
        current_route.append(self.depot)
        total_distance += route_distance
        routes.append(current_route)

        return total_distance, routes
      
class Solver:
    def __init__(self, objective_function=None, num_iterations=None):
        self.objective_function = objective_function
        self.num_iterations = num_iterations
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.global_best_routes = []
        self.fitness_history = []
    
    def optimize(self, verbose=True) -> tuple[list[any], float]:
        raise NotImplementedError
    
    def plot_fitness_history(self):
        """
        Plots the best fitness value over iterations.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.num_iterations + 1), self.fitness_history, marker='o', linestyle='-', color='b')
        plt.title('Fitness History')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness')
        plt.grid(True)
        plt.show()

    def plot_search_space(self):
        """
        Plots the contour of the 2D search space along with the best solution.
        """
        if len(self.search_space) != 2:
            print("Search space must be 2D for visualization.")
            return

        x = np.linspace(self.search_space[0][0], self.search_space[0][1], 100)
        y = np.linspace(self.search_space[1][0], self.search_space[1][1], 100)
        X, Y = np.meshgrid(x, y)
        
        # Compute the objective function value for each (x, y)
        Z = np.array([[self.objective_function([xi, yi]) for xi, yi in zip(x_row, y_row)] 
                      for x_row, y_row in zip(X, Y)])
        
        plt.figure(figsize=(10, 6))
        plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar(label='Fitness')
        if self.global_best_position:
            plt.scatter(self.global_best_position[0], self.global_best_position[1], color='red', label='Best Solution')
        plt.title('Search Space and Best Solution')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()

    def plot_routes(self, routes: list[list[Node]] = None) -> None:
        """
        Plot the routes for the VRP.
        
        Parameters:
            routes (list[list[Node]]): A list of routes, where each route is a list of Node objects.
                                    It is assumed that each route starts and ends at the depot.
        """
        plt.figure(figsize=(10, 8))
        
        # Define a set of colors to cycle through for different routes.
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange']
        if routes is None:
            routes = self.global_best_routes
            
        for i, route in enumerate(routes):
            # Extract the x and y coordinates for each node in the route.
            xs = [node.x for node in route]
            ys = [node.y for node in route]
            color = colors[i % len(colors)]
            
            # Plot the route as a line connecting the nodes.
            plt.plot(xs, ys, marker='o', linestyle='-', color=color, label=f"Route {i+1}")
            
            # Optionally, annotate each node with its node_id.
            for node in route:
                plt.text(node.x, node.y, f"{node.node_id}", fontsize=9, color=color,
                        verticalalignment='bottom', horizontalalignment='right')
        
        plt.title("VRP Routes")
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.legend()
        plt.grid(True)
        plt.show()

class Experiment:
    def __init__(self, instance, solvers, num_experiments=50):
        self.instance = instance
        self.solvers: list[Solver] = []
        self.names = []
        for s, name in solvers:
            self.solvers.append(s)
            self.names.append(name)
        self.num_experiments = num_experiments
        self.results: dict = None

    def run(self):
        results = {s: {
            "name": name,
            "fitness": [],
            "time": []
        } for s, name in zip(self.solvers, self.names)}
        for _ in range(self.num_experiments):
            print(f"Experiment {_ + 1}")
            for s in self.solvers:
                print("\t", results[s]["name"], end=": ")
                start = time.time()
                s.optimize(verbose=False)
                end = time.time()
                run_time = end - start
                results[s]["fitness"].append(s.global_best_fitness)
                results[s]["time"].append(run_time)
                print(f"Done: Fitness = {s.global_best_fitness}, Time = {run_time:.2f} sec")
        self.results = results
        return results
    
    def report(self):
        if self.results is None:
            print("No results to report. Run the experiments first.")
            return
        
        for s, r in self.results.items():
            print(f"\nSolver: {r['name']}")
            print(f"Mean Fitness: {np.mean(r['fitness'])}")
            print(f"Std Fitness: {np.std(r['fitness'])}")
            print(f"Mean Time: {np.mean(r['time'])}")
            print(f"Std Time: {np.std(r['time'])}")
            print("-"*20)

def print_routes(routes: list[list[Node]]):
    print("Number of routes: ", len(routes))
    print("Longest route: ", max([len(route) for route in routes]))
    print("Shortest route: ", min([len(route) for route in routes]))
    for route in routes:
        for node in route:
            print(node.model_dump()["node_id"], end=" ")
        print()
        
if __name__ == "__main__":
    instance = Problem()
    instance.load_data("data/25/C101_co_25.txt")

    initial_position = instance.initialize_position(p=0.5)
    print(len(initial_position))
    print("Initial Position: ", initial_position)
    cost, routes = instance.position2route(initial_position)
    print("Initial Cost: ", cost)
    print_routes(routes)
    
