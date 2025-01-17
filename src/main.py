from utils import load_data
from utils.config import DATA_PATH

loader = load_data.VRPDataLoader(DATA_PATH)
loader.load_data()

print("Number of Customers:", loader.get_num_customers())
print("Number of Lockers:", loader.get_num_lockers())
print("Number of Vehicles:", loader.get_num_vehicles())
print("Vehicle Capacity:", loader.get_capacity())
print("Demands:", loader.get_demands())
for data in loader.get_customer_data():
    print("\t", data)
print("Depot Data:", loader.get_depot_data())
print("Locker Data:", loader.get_locker_data())
print("Locker Assignments:", loader.get_locker_assignments())

# Display the distance matrix
distance_matrix = loader.get_distance_matrix()
print("Distance Matrix:")
for row in distance_matrix:
    print("\t", row)
