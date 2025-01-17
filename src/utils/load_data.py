from pydantic import BaseModel
from typing import List, Optional

class DataPoint(BaseModel):
    x: float
    y: float
    earliest: int
    latest: int
    service_time: int
    
class CustomerDataPoint(DataPoint):
    customer_type: int
    assigned_lockers: Optional[List[int]] = None

class LockerDataPoint(DataPoint):
    pass

class DepotDataPoint(DataPoint):
    pass

class VRPDataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.num_customers = 0
        self.num_lockers = 0
        self.num_vehicles = 0
        self.capacity = 0
        self.demands: List[int] = []
        self.customer_data: List[CustomerDataPoint] = []
        self.locker_assignments: List[List[int]] = []
        self.customer_types: List[int] = []
        self.depot_data: List[DepotDataPoint] = []
        self.locker_data: List[LockerDataPoint] = []
        self.distance_matrix = None

    def load_data(self):
        with open(self.file_path, 'r') as file:
            lines = file.readlines()
            # Extract number of customers and lockers
            self.num_customers, self.num_lockers = map(int, lines[0].split())
            
            # Extract number of vehicles and vehicle capacity
            self.num_vehicles, self.capacity = map(int, lines[1].split())
            
            # Extract demands
            self.demands = [int(lines[i].strip()) for i in range(2, 2 + self.num_customers)]
            
            # Extract depot data point, customer data points, locker data points and locker assignments
            # Depot data point is the first data point
            # Customer data points are the next num_customers data points
            # Locker data points are the next num_lockers data points
            # Locker assignments are the next num_customers data points

            # Extract depot data point
            current_index = 2 + self.num_customers
            # While the line has 5 elements, it is a data point
            while len(lines[current_index].split()) == 6:
                x, y, earliest, latest, service_time, type = map(int, lines[current_index].split())
                if type == 0:
                    self.depot_data.append(DepotDataPoint(x=x, y=y, earliest=earliest, latest=latest, service_time=service_time))
                elif type == 4:
                    self.locker_data.append(LockerDataPoint(x=x, y=y, earliest=earliest, latest=latest, service_time=service_time))
                else:
                    self.customer_data.append(CustomerDataPoint(x=x, y=y, earliest=earliest, latest=latest, service_time=service_time, customer_type=type))
                current_index += 1
            # Extract locker assignments
            for index, i in enumerate(range(current_index, current_index + self.num_customers)):
                assigned_lockers = list(map(int, lines[i].split()))
                self.customer_data[index].assigned_lockers = assigned_lockers
                self.locker_assignments.append(list(map(int, lines[i].split())))

    def get_distance_matrix(self):
        if self.distance_matrix is not None:
            return self.distance_matrix
        distances = []
        for i in range(len(self.customer_data)):
            row = []
            for j in range(len(self.customer_data)):
                row.append(((self.customer_data[i].x - self.customer_data[j].x) ** 2 + (self.customer_data[i].y - self.customer_data[j].y) ** 2) ** 0.5)
            distances.append(row)
        self.distance_matrix = distances
        return distances

    def get_demands(self):
        return self.demands

    def get_customer_data(self):
        return self.customer_data

    def get_capacity(self):
        return self.capacity

    def get_num_customers(self):
        return self.num_customers

    def get_num_lockers(self):
        return self.num_lockers

    def get_num_vehicles(self):
        return self.num_vehicles

    def get_customer_types(self):
        return self.customer_types

    def get_locker_assignments(self):
        return self.locker_assignments

    def get_depot_data(self):
        return self.depot_data

    def get_locker_data(self):
        return self.locker_data

# Example usage:
# loader = VRPDataLoader('/mnt/data/C101_co_25.txt')
# loader.load_data()
# print("Number of Customers:", loader.get_num_customers())
# print("Number of Lockers:", loader.get_num_lockers())
# print("Number of Vehicles:", loader.get_num_vehicles())
# print("Vehicle Capacity:", loader.get_capacity())
# print("Demands:", loader.get_demands())
# print("Customer Data:", loader.get_customer_data())
# print("Depot Data:", loader.get_depot_data())
# print("Locker Data:", loader.get_locker_data())
# print("Locker Assignments:", loader.get_locker_assignments())
