# ...existing code...

from dataclasses import dataclass
from typing import List, Union
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class VehicleInstance:
    def __init__(self, number, capacity):
        self.number = number
        self.capacity = capacity

@dataclass
class Vehicle:
    _id: int
    capacity: int
    time_limit: int
    fixed_cost: int
    variable_cost: int
    start: int
    end: int

class CustomerInstance:
    def __init__(self, cust_no, xcoord, ycoord, demand, ready_time, due_date, service_time):
        self.cust_no = cust_no
        self.xcoord = xcoord
        self.ycoord = ycoord
        self.demand = demand
        self.ready_time = ready_time
        self.due_date = due_date
        self.service_time = service_time
    
    def __repr__(self) -> str:
        print(f"Customer: {self.cust_no}, {self.xcoord}, {self.ycoord}, {self.demand}, {self.ready_time}, {self.due_date}, {self.service_time}")

@dataclass
class Location:
    id: int
    x: int
    y: int
    ready_time: int = 0
    due_time: int = 0
    service_time: int = 0

class OrderItem:
    def __init__(self, order_id, start_location: Location, end_location: Location, demand: int, ready_time: int, due_date: int, service_time: int):
        self.order_id = order_id
        self.start_location = start_location
        self.end_location = end_location
        self.demand = demand
        self.ready_time = ready_time
        self.due_date = due_date
        self.service_time = service_time
        self.distance = self.get_distance()

    def get_distance(self):
        return ((self.start_location.x - self.end_location.x) ** 2 + (self.start_location.y - self.end_location.y) ** 2) ** 0.5

    def __repr__(self):
        return f"Order: {self.start_location.id} -> {self.end_location.id}: demand={self.demand}, distance={self.distance:.2f}"

def load_solomon_vrp(file_path) -> Union[VehicleInstance, List[CustomerInstance]]:
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    vehicle_info: VehicleInstance = None
    customer_data: List[CustomerInstance] = []
    section = None    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('VEHICLE'):
            section = 'vehicle'
            continue
        elif line.startswith('CUSTOMER'):
            section = 'customer'
            continue
        if section == 'vehicle':
            if 'NUMBER' in line and 'CAPACITY' in line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                vehicle_info = VehicleInstance(int(parts[0]), int(parts[1]))
        elif section == 'customer':
            if line.startswith('CUST NO.'):
                continue
            parts = line.split()
            if len(parts) == 7:
                customer = CustomerInstance(int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6]))
                customer_data.append(customer)
    
    return vehicle_info, customer_data

def load_voratas_vrp(file_path, number_instance) -> Union[List[OrderItem], List[Location], List[Vehicle]]:
    with open(file_path, 'r') as f:
        lines = f.readlines()

    locations: List[Location] = []
    orders: List[OrderItem] = []
    vehicles: List[Vehicle] = []
    order_matrix = np.zeros((number_instance + 1, number_instance + 1))

    line_no = 0
    for line in lines:
        line_no += 1
        line = line.strip()
        if not line:
            continue
        # Load location and order matrix
        if line_no >= 12 and line_no <= 32:
            parts = line.split()
            if len(parts) == 30:
                location = Location(
                    id=int(parts[0]),
                    x=int(parts[1]),
                    y=int(parts[2]),
                    ready_time=int(parts[3]),
                    due_time=int(parts[4]),
                    service_time=int(parts[5])
                )
                locations.append(location)
                for index, part in enumerate(parts[9:]):
                    assert line_no >= 12 and line_no <= 32
                    assert index >= 0 and index <= number_instance, f"Index: {index}"
                    order_matrix[line_no-12][index] = int(part)
        # Load vehicle information
        if line_no >= 43 and line_no <= 58:
            parts = line.split()
            if len(parts) == 7:
                vehicle = Vehicle(
                    _id=int(parts[0]),
                    capacity=int(parts[1]),
                    time_limit=int(parts[2]),
                    fixed_cost=int(parts[3]),
                    variable_cost=int(parts[4]),
                    start=int(parts[5]),
                    end=int(parts[6])
                )
                vehicles.append(vehicle)
                
    for i in range(order_matrix.shape[0]):
        for j in range(order_matrix.shape[1]):
            if i != j and order_matrix[i][j] > 0:
                start_location = locations[i]
                end_location = locations[j]
                order = OrderItem(i, start_location, end_location, order_matrix[i][j], start_location.ready_time, start_location.due_time, start_location.service_time)
                orders.append(order)

    return orders, locations, vehicles

def convert_to_orders(customer_data: List[CustomerInstance]) -> List[OrderItem]:
    """
    Convert the customer data to a list of orders, with depot
    """
    orders = []
    start_location = Location(0, customer_data[0].xcoord, customer_data[0].ycoord)
    for customer in customer_data[1:]:
        end_location = Location(customer.cust_no, customer.xcoord, customer.ycoord)
        order = OrderItem(customer.cust_no, start_location, end_location, customer.demand, customer.ready_time, customer.due_date, customer.service_time)
        orders.append(order)
    return orders

def visualize_orders(orders: List[OrderItem], title="Orders"):
    G = nx.DiGraph()
    for order in orders:
        pickup = order.start_location.id
        delivery = order.end_location.id
        G.add_edge(pickup, delivery)
    pos = nx.spring_layout(G, k=0.8)  # Increase the value of k to increase the distance between nodes
    nx.draw(G, pos, with_labels=True, arrows=True, node_size=50)
    plt.title(title)
    plt.savefig('output/orders.png')
    print("Orders visualized in output/orders.png")
    # plt.show()