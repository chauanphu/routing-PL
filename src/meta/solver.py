import math
import time
from typing import List
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
    assigned_locker: int | None = None

class Vehicle(BaseModel):
    """
    VRPPL Vehicle class

    Storing the vehicle data including the vehicle id, capacity, and fixed cost
    """
    v_id: int
    capacity: float
    load: float
    current_time: float

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
                self.lockers.append(Node(node_id=i, x=x, y=y, early=early, late=late, service=service))
            # Read locker assignment data
            for i in range(self.num_customers):
                for j, assignment in enumerate(map(int, f.readline().split())):
                    if assignment:
                        self.customers[i].assigned_locker = j + 1
                        break

    @staticmethod
    def euclidean_distance(n1: Node, n2: Node) -> float:
        """Helper function to calculate Euclidean distance between two nodes."""
        return math.hypot(n1.x - n2.x, n1.y - n2.y)

    def get_search_space(self) -> list[tuple[float, float]]:
        """
        Get the search space for the problem.

        Returns:
            search_space (list[tuple[float, float]]): The search space for the problem.
        """
        return [(0.0, 1.0) for _ in range(self.num_customers)]

    def evaluate_position(self, positions: list[float]) -> float:
        """
        Decode the positions (continuous values) to a permutation of customer visits,
        split the permutation into routes for vehicles while checking feasibility,
        and compute the total travel distance.
        
        Parameters:
            positions (list[float]): a list of continuous values associated with each customer.
            
        Returns:
            total_distance (float): total distance traveled by all vehicles. If infeasible, returns a high penalty.
        """
        # ------------------ Step 1: Decode Positions to a Customer Permutation ------------------
        # Assume positions list is of length equal to the number of customers.
        # Sort customer indices based on the continuous values.
        sorted_indices = sorted(range(len(positions)), key=lambda i: positions[i])
        # Create an ordered list of customers to be visited.
        ordered_customers = [self.customers[i] for i in sorted_indices]

        # ------------------ Step 2: Split the Permutation into Routes ------------------
        total_distance = 0.0
        vehicle_index = 0  # index for the current vehicle in self.vehicles
        
        # Initialize the first vehicle's state (start at the depot with zero load/time)
        current_vehicle = Vehicle(
            v_id=vehicle_index,
            capacity=self.vehicle_capacity,
            load=0.0,
            current_time=0.0
        )
        # Start the route from the depot.
        last_node = self.depot
        route_distance = 0.0

        # Loop through each customer in the ordered permutation.
        for customer in ordered_customers:
            # Calculate travel time from the last node to the customer.
            travel_time = self.euclidean_distance(last_node, customer)
            arrival_time = current_vehicle.current_time + travel_time

            # Respect the customer's time window: wait if arriving too early.
            if arrival_time < customer.early:
                arrival_time = customer.early

            # Check if the customer can be feasibly served by the current vehicle.
            # Two checks: (i) Time window feasibility and (ii) Vehicle capacity.
            if (arrival_time > customer.late) or (current_vehicle.load + customer.demand > current_vehicle.capacity):
                # End the current route: add the return trip to the depot.
                route_distance += self.euclidean_distance(last_node, self.depot)
                total_distance += route_distance

                # Move to the next available vehicle.
                vehicle_index += 1
                if vehicle_index >= self.num_vehicles:
                    # If no more vehicles are available, return a penalty cost (infeasible solution).
                    return float('inf')
                
                # Reinitialize the new vehicle from the depot.
                current_vehicle = Vehicle(
                    v_id=vehicle_index,
                    capacity=self.vehicle_capacity,
                    load=0.0,
                    current_time=0.0
                )
                # Reset the route distance and last node.
                route_distance = 0.0
                last_node = self.depot

                # Recalculate travel details from the depot to the current customer.
                travel_time = self.euclidean_distance(last_node, customer)
                arrival_time = current_vehicle.current_time + travel_time
                if arrival_time < customer.early:
                    arrival_time = customer.early

                # If even after starting from the depot the customer cannot be served, mark as infeasible.
                if (arrival_time > customer.late) or (current_vehicle.load + customer.demand > current_vehicle.capacity):
                    return float('inf')
            
            # ------------------ Step 3: Add Customer to the Current Route ------------------
            # Add the travel time from last node to the customer.
            route_distance += travel_time
            # Update the vehicle's current time (arrival plus service time at the customer).
            current_vehicle.current_time = arrival_time + customer.service
            # Add the customer's demand to the vehicle's load.
            current_vehicle.load += customer.demand

            # Set the current customer as the last node for the next iteration.
            last_node = customer

        # After visiting all customers, finish the last route by returning to the depot.
        route_distance += self.euclidean_distance(last_node, self.depot)
        total_distance += route_distance

        # ------------------ Step 4: Return the Objective Value ------------------
        return total_distance

    def position2route(self, positions: list[float]) -> tuple[float, list[list[Node]]]:
        """
        Decode the positions (continuous values) to a permutation of customer visits,
        split the permutation into routes for vehicles while checking feasibility,
        and compute the total travel distance along with the routes (as lists of Node).
        
        Parameters:
            positions (list[float]): a list of continuous values associated with each customer.
            
        Returns:
            A tuple (total_distance, routes) where:
            - total_distance (float): total distance traveled by all vehicles. 
                Returns float('inf') if the solution is infeasible.
            - routes (list[list[Node]]): a list of routes, where each route is a list 
                of Node objects (starting and ending at the depot).
        """
        # ------------------ Step 1: Decode Positions to a Customer Permutation ------------------
        # Assume positions list is of length equal to the number of customers.
        # Sort customer indices based on the continuous values.
        sorted_indices = sorted(range(len(positions)), key=lambda i: positions[i])
        # Create an ordered list of customers to be visited.
        ordered_customers = [self.customers[i] for i in sorted_indices]

        # ------------------ Step 2: Split the Permutation into Routes ------------------
        total_distance = 0.0
        routes = []  # List to store the routes (each route is a list of Node)
        vehicle_index = 0  # index for the current vehicle in self.vehicles

        # Initialize the first vehicle's state (start at the depot with zero load/time)
        current_vehicle = Vehicle(
            v_id=vehicle_index,
            capacity=self.vehicle_capacity,
            load=0.0,
            current_time=0.0
        )
        # Start the current route from the depot.
        current_route = [self.depot]
        last_node = self.depot
        route_distance = 0.0

        # Loop through each customer in the ordered permutation.
        for customer in ordered_customers:
            # Calculate travel time from the last node to the customer.
            travel_time = self.euclidean_distance(last_node, customer)
            arrival_time = current_vehicle.current_time + travel_time

            # Respect the customer's time window: wait if arriving too early.
            if arrival_time < customer.early:
                arrival_time = customer.early

            # Check if the customer can be feasibly served by the current vehicle.
            # Two checks: (i) Time window feasibility and (ii) Vehicle capacity.
            if (arrival_time > customer.late) or (current_vehicle.load + customer.demand > current_vehicle.capacity):
                # End the current route: add the return trip to the depot.
                return_to_depot = self.euclidean_distance(last_node, self.depot)
                route_distance += return_to_depot
                current_route.append(self.depot)  # finish route at depot
                total_distance += route_distance
                routes.append(current_route)

                # Move to the next available vehicle.
                vehicle_index += 1
                if vehicle_index >= self.num_vehicles:
                    # If no more vehicles are available, return a penalty cost (infeasible solution).
                    return float('inf'), []
                
                # Reinitialize the new vehicle from the depot.
                current_vehicle = Vehicle(
                    v_id=vehicle_index,
                    capacity=self.vehicle_capacity,
                    load=0.0,
                    current_time=0.0
                )
                # Reset the route details.
                route_distance = 0.0
                current_route = [self.depot]
                last_node = self.depot

                # Recalculate travel details from the depot to the current customer.
                travel_time = self.euclidean_distance(last_node, customer)
                arrival_time = current_vehicle.current_time + travel_time
                if arrival_time < customer.early:
                    arrival_time = customer.early

                # If even after starting from the depot the customer cannot be served, mark as infeasible.
                if (arrival_time > customer.late) or (current_vehicle.load + customer.demand > current_vehicle.capacity):
                    return float('inf'), []
            
            # ------------------ Step 3: Add Customer to the Current Route ------------------
            # Add the travel time from last node to the customer.
            route_distance += travel_time
            current_route.append(customer)
            # Update the vehicle's current time (arrival plus service time at the customer).
            current_vehicle.current_time = arrival_time + customer.service
            # Add the customer's demand to the vehicle's load.
            current_vehicle.load += customer.demand

            # Set the current customer as the last node for the next iteration.
            last_node = customer

        # After visiting all customers, finish the last route by returning to the depot.
        return_to_depot = self.euclidean_distance(last_node, self.depot)
        route_distance += return_to_depot
        current_route.append(self.depot)
        total_distance += route_distance
        routes.append(current_route)

        # ------------------ Step 4: Return the Objective Value and Routes ------------------
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

    def random_route(self) -> list[Node]:
        """
        Generate a random route that includes all customers in a random order,
        with the depot (node with _id 0) at the start and at the end.

        Returns:
            route (list[Node]): A list of Node objects representing the route.
                                The first and last nodes are the depot.
        """
        # Start the route with the depot.
        route = [self.depot]
        
        # Create a copy of the customers list and shuffle it.
        customers_copy = self.customers.copy()
        random.shuffle(customers_copy)
        
        # Append the randomly ordered customers.
        route.extend(customers_copy)
        
        # End the route with the depot.
        route.append(self.depot)
        
        return route

    def permu2route(self, permutation: list[Node | Customer], contain_depot=True) -> tuple[float, list[list[Node]]]:
        """
        Decode a permutation of customers into routes using a cumulative criterion.
        
        The cumulative measure used here is the sum of customer demands.
        A route is built by adding customers one by one until adding the next customer 
        would exceed the vehicle capacity. At that point, the current route is terminated 
        (by appending the depot) and a new route is started.
        
        The depot (self.depot) is added at the start and end of each route.
        
        Parameters:
            permutation (list[Node]): A list of customer Node objects (without depot markers)
                                        representing a permutation of visits.
        
        Returns:
            routes (list[list[Node]]): A list of routes, each route being a list of Node objects,
                                       with the depot at the beginning and end.
        
        Example:
            Suppose permutation (customer IDs) is [4, 7, 14] and using cumulative demand, a split is made
            after the first customer. Then the decoded routes will be:
                [0, 4, 0] and [0, 7, 14, 0],
            where 0 represents the depot node.
        """
        routes = []
        current_route = [self.depot]  # start route with depot
        # Ignore the depots if the permutation does not contain it.
        if contain_depot:
            permutation = permutation[1:-1]

                # ------------------ Step 2: Split the Permutation into Routes ------------------
        total_distance = 0.0
        vehicle_index = 0  # index for the current vehicle in self.vehicles
        
        # Initialize the first vehicle's state (start at the depot with zero load/time)
        current_vehicle = Vehicle(
            v_id=vehicle_index,
            capacity=self.vehicle_capacity,
            load=0.0,
            current_time=0.0
        )
        # Start the route from the depot.
        last_node = self.depot
        route_distance = 0.0

        # Loop through each customer in the ordered permutation.
        for customer in permutation:
            # Calculate travel time from the last node to the customer.
            travel_time = self.euclidean_distance(last_node, customer)
            arrival_time = current_vehicle.current_time + travel_time

            # Respect the customer's time window: wait if arriving too early.
            if arrival_time < customer.early:
                arrival_time = customer.early
                
            demand = customer.demand if hasattr(customer, "demand") else 0
            # Check if the customer can be feasibly served by the current vehicle.
            # Two checks: (i) Time window feasibility and (ii) Vehicle capacity.
            if (arrival_time > customer.late) or (current_vehicle.load + demand > current_vehicle.capacity):
                # End the current route: add the return trip to the depot.
                return_to_depot = self.euclidean_distance(last_node, self.depot)
                route_distance += return_to_depot
                current_route.append(self.depot)  # finish route at depot
                total_distance += route_distance
                routes.append(current_route)

                # Move to the next available vehicle.
                vehicle_index += 1
                if vehicle_index >= self.num_vehicles:
                    # If no more vehicles are available, return a penalty cost (infeasible solution).
                    return float('inf'), []
                
                # Reinitialize the new vehicle from the depot.
                current_vehicle = Vehicle(
                    v_id=vehicle_index,
                    capacity=self.vehicle_capacity,
                    load=0.0,
                    current_time=0.0
                )
                # Reset the route details.
                route_distance = 0.0
                current_route = [self.depot]
                last_node = self.depot

                # Recalculate travel details from the depot to the current customer.
                travel_time = self.euclidean_distance(last_node, customer)
                arrival_time = current_vehicle.current_time + travel_time
                if arrival_time < customer.early:
                    arrival_time = customer.early

                # If even after starting from the depot the customer cannot be served, mark as infeasible.
                if (arrival_time > customer.late) or (current_vehicle.load + demand > current_vehicle.capacity):
                    return float('inf'), []
            
            # ------------------ Step 3: Add Customer to the Current Route ------------------
            # Add the travel time from last node to the customer.
            route_distance += travel_time
            current_route.append(customer)
            # Update the vehicle's current time (arrival plus service time at the customer).
            current_vehicle.current_time = arrival_time + customer.service
            # Add the customer's demand to the vehicle's load.
            current_vehicle.load += demand

            # Set the current customer as the last node for the next iteration.
            last_node = customer

        # After visiting all customers, finish the last route by returning to the depot.
        return_to_depot = self.euclidean_distance(last_node, self.depot)
        route_distance += return_to_depot
        current_route.append(self.depot)
        total_distance += route_distance
        routes.append(current_route)

        # ------------------ Step 4: Return the Objective Value and Routes ------------------
        return total_distance, routes
      
class Solver:
    def __init__(self, objective_function=None, num_iterations=None, search_space=None):
        self.objective_function = objective_function
        self.num_iterations = num_iterations
        self.search_space = search_space
        self.global_best_position = None
        self.global_best_fitness = float('inf')
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
    print("Number of customers: ", instance.num_customers)
    print("Number of lockers: ", instance.num_lockers)
    print("Number of vehicles: ", instance.num_vehicles)
    print("Vehicle capacity: ", instance.vehicle_capacity)
    print("First 5 customers")
    for customer in instance.customers[:5]:
        print("\t",customer)
    print(f"First {instance.num_lockers if instance.num_lockers < 5 else 5} lockers:")
    for locker in instance.lockers[:5]:
        print("\t",locker)

    initial_solution = instance.random_route()
    print("Initial solution:")
    print_routes([initial_solution])
    total_cost, routes = instance.permu2route(initial_solution)
    print("Initial solution cost: ", total_cost)
    print("Initial solution routes:")
    print_routes(routes)