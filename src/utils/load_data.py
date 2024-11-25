# ...existing code...

from typing import List, Union

class VehicleInstance:
    def __init__(self, number, capacity):
        self.number = number
        self.capacity = capacity

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

class Location:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.isDepot = False

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
