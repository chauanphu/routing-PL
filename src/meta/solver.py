import math
from pydantic import BaseModel

class Node(BaseModel):
    """
    VRPPL Node class: customers, depots, and lockers

    Storing the node data including the node id, x and y coordinates
    """
    _id: int
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
    _id: int
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
            self.depot = Node(_id=0, x=x, y=y, early=early, late=late, service=service)
            
            # Read customer data
            self.customers = []
            for i in range(1, self.num_customers + 1):
                x, y, early, late, service, customer_type = map(float, f.readline().split())
                self.customers.append(Customer(_id=i, x=x, y=y, early=early, late=late, service=service, demand=demands[i-1], customer_type=customer_type))
            # Read locker data
            self.lockers = []
            for i in range(1, self.num_lockers + 1):
                x, y, early, late, service, _ = map(float, f.readline().split())
                self.lockers.append(Node(_id=i, x=x, y=y, early=early, late=late, service=service))
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

    def evaluate(self, positions: list[float]) -> float:
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
            _id=vehicle_index,
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
                    _id=vehicle_index,
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

    # Generate random positions for the customers
    import random
    positions = [random.random() for _ in range(instance.num_customers)]
    print("Random positions: ", positions)
    total_distance = instance.evaluate(positions)
    print("Total distance: ", total_distance)

    