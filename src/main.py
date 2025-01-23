from matplotlib import pyplot as plt
from meta.solver import Problem

# Create a problem instance
instance = Problem()
instance.load_data("data/25/C101_co_25.txt")
print("Number of customers: ", instance.num_customers)
print("Number of lockers: ", instance.num_lockers)
print("Number of vehicles: ", instance.num_vehicles)
print("Vehicle capacity: ", instance.vehicle_capacity)

# Print sample data
# First 5 customers
print("First 5 customers")
print(instance.customers[:5])
# First 5 lockers
print("First 5 lockers")
print(instance.lockers[:5])
