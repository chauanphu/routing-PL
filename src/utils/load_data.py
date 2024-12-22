# ...existing code...

from dataclasses import dataclass
from typing import List, Union
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

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
        self.due_time = due_date
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

def load_voratas_vrp(instance) -> Union[List[OrderItem], List[Location], List[Vehicle]]:
    file_path = f'benchmark/voratas/{instance}/'
    demand_file = file_path + 'demands.txt'
    vehicles = file_path + 'vehicles.txt'
    # 0: id, 1: x, 2: y, 3: ready_time, 4: due_time, 5: service_time, 6: variable service time, 7: allow early, 8: allow late
    demand_df = pd.read_csv(demand_file, sep='\t', header=None)
    vehicles_df = pd.read_csv(vehicles, sep='\t', header=None)
    locations_df = demand_df.iloc[:, 0:9]
    demand_df = demand_df.iloc[:, 9:]
    num_locations = len(demand_df)
    locations: List[Location] = []
    orders: List[OrderItem] = []
    vehicles: List[Vehicle] = []

    for i in range(num_locations):
        location = Location(i, locations_df.iloc[i, 1], locations_df.iloc[i, 2], locations_df.iloc[i, 3], locations_df.iloc[i, 4], locations_df.iloc[i, 5])
        locations.append(location)

    order_id = 0
    for i in range(num_locations):
        for j in range(num_locations):
            if i == j:
                continue
            else:
                demand = demand_df.iloc[i, j]
                if demand == 0:
                    continue
                ready_time = demand_df.iloc[j, 3]
                due_time = demand_df.iloc[j, 4]
                service_time = demand_df.iloc[j, 5]
                order = OrderItem(order_id, locations[i], locations[j], demand, ready_time, due_time, service_time)
                orders.append(order)
                order_id += 1

    for i in range(len(vehicles_df)):
        vehicle = Vehicle(i, vehicles_df.iloc[i, 1], vehicles_df.iloc[i, 2], vehicles_df.iloc[i, 3], vehicles_df.iloc[i, 4], vehicles_df.iloc[i, 5], vehicles_df.iloc[i, 6])
        vehicles.append(vehicle)

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

def visualize_orders(orders: List[OrderItem], vehicles: List[Vehicle], title: str, output_name: str = ''):
    G = nx.DiGraph()
    vehicle_station: List[int] = [vehicle.start for vehicle in vehicles]
    vehicle_station = set(vehicle_station)

    for order in orders:
        G.add_node(order.start_location.id, pos=(order.start_location.x, order.start_location.y))
        G.add_node(order.end_location.id, pos=(order.end_location.x, order.end_location.y))
        G.add_edge(order.start_location.id, order.end_location.id, weight=order.distance)
    
    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(10, 10))
    plt.title(title)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_weight='bold')
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=7)

    # Highlight vehicle stations in red
    nx.draw_networkx_nodes(G, pos, nodelist=vehicle_station, node_color='red', node_size=500)

    plt.savefig(f'output/{output_name}.png')
    print("Orders visualized in output/{output_name}.png")
    # plt.show()