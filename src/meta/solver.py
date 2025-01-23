from typing import List
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
    vehicles: list[Vehicle]
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