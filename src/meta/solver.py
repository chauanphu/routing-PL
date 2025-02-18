import math
import timeit
from typing import Any, List, Tuple, Optional
import numpy as np
from pydantic import BaseModel
from matplotlib import pyplot as plt
import random
import os

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
    assigned_locker: Optional[Node] = None  # changed union operator to Optional
    locker_delivery: Optional[bool] = None    # changed union operator to Optional

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
    locker_cache = {}

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
        print("Loading data from", path)
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
        print("Data loaded successfully")

    @staticmethod
    def euclidean_distance(n1: Node, n2: Node) -> float:
        """Helper function to calculate Euclidean distance between two nodes."""
        return math.hypot(n1.x - n2.x, n1.y - n2.y)
    
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

    def initialize_solution(self, p: float) -> Tuple[List[int], List[Node]]:
        # Copy of customers to be assigned.
        sorted_customers = sorted(self.customers, key=lambda cust: cust.node_id)
        # For each customer, assign a delivery node based on type.
        # For type 1, assign the customer itself.
        # For type 2 and type 3, assign the selected (nearest) parcel locker station.
        assignment_list = []
        for customer in sorted_customers:
            chosen_node, _ = self._locker_assignment(customer, explore=True, p=p)
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
        
        return first_row, second_row
    
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


    def _locker_assignment(self, customer: Customer, explore=False, p=0.5) -> tuple[Node, Customer]:
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
        elif customer.customer_type == 3:
            if not explore:
                chosen_node = self.locker_cache.get(customer.node_id)
                if chosen_node is not None:
                    return chosen_node, customer
            if random.random() < p: # If r < p then home delivery
                chosen_node = customer
                customer.locker_delivery = False
                self.locker_cache[customer.node_id] = customer
            else:
                chosen_node = customer.assigned_locker
                customer.locker_delivery = True
                self.locker_cache[customer.node_id] = customer.assigned_locker
        else: # In case 
            chosen_node = customer
        return chosen_node, customer

    def permu2route(self, permutation: list[int]) -> tuple[float, list[list[Node]]]:
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
            chosen_node, customer = self._locker_assignment(customer, False)

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

    def node2routes(self, permutation: list[Node]) -> tuple[float, list[list[Node]]]:
        """
        Evaluate a permutation of node IDs (integers) that includes both customer and locker nodes
        and convert it into feasible routes. The permutation is expected to include the depot
        (node with id 0) as the first and last elements.

        Parameters:
            permutation (list[int]): A list of node IDs representing the visiting order.
        
        Returns:
            tuple: (total_distance, routes)
                total_distance (float): The sum of the travel distances for all routes. If any route is
                                        infeasible, returns float('inf').
                routes (list[list[Node]]): A list of routes, where each route is a list of Node objects
                                        (starting and ending with the depot).
        """
        routes = []
        total_distance = 0.0

        # Initialize the first route starting at the depot.
        current_route = [self.depot]
        vehicle_index = 0
        current_vehicle = Vehicle(
            v_id=vehicle_index, 
            capacity=self.vehicle_capacity, 
            load=0.0, 
            current_time=0.0
        )
        last_node = self.depot
        route_distance = 0.0
        visited: set[int] = set()

        # Build a lookup dictionary for all nodes by their node_id.
        node_dict = {}
        node_dict[self.depot.node_id] = self.depot
        for cust in self.customers:
            node_dict[cust.node_id] = cust
        for locker in self.lockers:
            node_dict[locker.node_id] = locker

        # Process each node ID from the permutation, skipping the first and last (depot markers)
        for nid in permutation[1:-1]:
            node: Node = node_dict.get(nid.node_id)
            if node is None:
                # If the node ID is not found, skip it.
                continue

            # --- Feasibility Check ---
            travel_time = self.euclidean_distance(last_node, node)
            arrival_time = current_vehicle.current_time + travel_time
            if arrival_time < node.early:
                arrival_time = node.early

            # Check for duplicate locker visits (if the same locker is visited non-consecutively).
            duplicate_lockers = (
                (node.node_id in visited) and 
                (node.node_id != self.depot.node_id) and 
                (node.node_id != last_node.node_id)
            )
            # For customer nodes, get the demand; for lockers or depot, assume zero demand.
            node_demand = node.demand if hasattr(node, "demand") else 0

            if (arrival_time > node.late) or (current_vehicle.load + node_demand > current_vehicle.capacity) or duplicate_lockers:
                # Terminate the current route by returning to the depot.
                return_to_depot = self.euclidean_distance(last_node, self.depot)
                route_distance += return_to_depot
                current_route.append(self.depot)
                total_distance += route_distance
                routes.append(current_route)

                # Switch to the next vehicle.
                vehicle_index += 1
                if vehicle_index >= self.num_vehicles:
                    # If no more vehicles are available, return an infeasible penalty.
                    return float('inf'), []
                current_vehicle = Vehicle(
                    v_id=vehicle_index, 
                    capacity=self.vehicle_capacity, 
                    load=0.0, 
                    current_time=0.0
                )
                # Reset route details.
                current_route = [self.depot]
                last_node = self.depot
                route_distance = 0.0
                visited.clear()

                # Recalculate travel details from the new route start (depot) to the current node.
                travel_time = self.euclidean_distance(last_node, node)
                arrival_time = current_vehicle.current_time + travel_time
                if arrival_time < node.early:
                    arrival_time = node.early
                if (arrival_time > node.late) or (current_vehicle.load + node_demand > current_vehicle.capacity):
                    return float('inf'), []

            # --- Accept the Node ---
            route_distance += travel_time
            current_route.append(node)
            current_vehicle.current_time = arrival_time + node.service
            current_vehicle.load += node_demand
            visited.add(node.node_id)
            last_node = node

        # Finalize the last route by returning to the depot.
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
        # Save the plot to a file
        plt.savefig("output/fitness_history.png")

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
        # Save the plot to a file
        plt.savefig("output/routes.png")

class Experiment:
    def __init__(self, instance, solvers, num_experiments=50):
        """
        Parameters:
            instance: The problem instance.
            solvers: A list of tuples (solver_instance, solver_name)
            num_experiments: Number of runs per solver.
        """
        self.instance = instance
        self.solvers: list[Solver] = []
        self.names = []
        for s, name in solvers:
            self.solvers.append(s)
            self.names.append(name)
        self.num_experiments = num_experiments
        self.results: dict = {}  # Will store results per solver

        # Create output directory if it does not exist.
        self.output_dir = "output/experiment"
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        """
        Runs each solver num_experiments times on the same dataset.
        Records the best fitness (distance) and execution time.
        """
        # Initialize results dictionary.
        results = {s: {
            "name": name,
            "fitness": [],
            "time": []
        } for s, name in zip(self.solvers, self.names)}
        
        for exp in range(self.num_experiments):
            print(f"Experiment {exp + 1}/{self.num_experiments}")
            for solver in self.solvers.copy():
                # Reset the solver's state before each run if needed.
                solver.global_best_fitness = float('inf')
                solver.global_best_position = None
                solver.global_best_routes = []
                solver.fitness_history = []

                print(f"\tRunning solver: {results[solver]['name']}", end="... ")
                run_time = timeit.timeit(lambda: solver.optimize(verbose=False), number=1)

                results[solver]["fitness"].append(solver.global_best_fitness)
                results[solver]["time"].append(run_time)
                print(f"Done: Fitness = {solver.global_best_fitness:.2f}, Time = {run_time:.2f} sec")
                del solver
        self.results = results
        return results

    def report(self):
        """
        Prints the performance report and exports the plots for each solver.
        The report includes the best distance, average distance, and standard deviation.
        The plots are saved to the output/experiment folder.
        """
        if not self.results:
            print("No results to report. Run the experiments first.")
            return

        for solver in self.solvers:
            r = self.results[solver]
            solver_name = r["name"]
            fitness_array = np.array(r["fitness"])
            best_distance = fitness_array.min()
            avg_distance = fitness_array.mean()
            std_distance = fitness_array.std()

            print(f"\nSolver: {solver_name}")
            print(f"Best Distance: {best_distance:.2f}")
            print(f"Average Distance: {avg_distance:.2f}")
            print(f"Std Deviation: {std_distance:.2f}")
            print("-" * 30)

            # Export fitness history plot (if available)
            if solver.fitness_history:
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(solver.fitness_history) + 1), solver.fitness_history,
                         marker='o', linestyle='-', color='b')
                plt.title(f'Fitness History - {solver_name}')
                plt.xlabel('Iteration')
                plt.ylabel('Best Fitness')
                plt.grid(True)
                fitness_plot_path = os.path.join(self.output_dir, f"fitness_history_{solver_name}.png")
                plt.savefig(fitness_plot_path)
                plt.close()
                print(f"Saved fitness history plot: {fitness_plot_path}")

            # Export routes plot (if available)
            if solver.global_best_routes:
                plt.figure(figsize=(10, 8))
                colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange']
                for i, route in enumerate(solver.global_best_routes):
                    xs = [node.x for node in route]
                    ys = [node.y for node in route]
                    color = colors[i % len(colors)]
                    plt.plot(xs, ys, marker='o', linestyle='-', color=color, label=f"Route {i+1}")
                    for node in route:
                        plt.text(node.x, node.y, f"{node.node_id}", fontsize=9, color=color,
                                 verticalalignment='bottom', horizontalalignment='right')
                plt.title(f"Routes - {solver_name}")
                plt.xlabel("X coordinate")
                plt.ylabel("Y coordinate")
                plt.legend()
                plt.grid(True)
                routes_plot_path = os.path.join(self.output_dir, f"routes_{solver_name}.png")
                plt.savefig(routes_plot_path)
                plt.close()
                print(f"Saved routes plot: {routes_plot_path}")

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

    initial_solution = instance.initialize_solution(0.5)
    print("First row:", [node for node in initial_solution[0]])
    print("Second row:", [node.node_id for node in initial_solution[1]])
    cost, routes = instance.permu2route(initial_solution[0])
    print("Cost:", cost)
    print_routes(routes)

