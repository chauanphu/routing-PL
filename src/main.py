from meta.PSO import MultiSwarmPSO
from utils import load_data
from utils.config import DATA_PATH

loader = load_data.VRPDataLoader(DATA_PATH)
loader.load_data()

# Customer data
customer_data = loader.get_customer_data()
# Vehicle data
vehicle_data = loader.get_num_vehicles()
# Depot data
depot_data = loader.get_depot_data()
print("Customer data")
for customer in customer_data:
    print(customer)
print("Vehicle data")
print(vehicle_data)
print("Depot data")
for depot in depot_data:
    print(depot)
# Print parcel locker data
parcel_locker_data = loader.locker_data
for locker in parcel_locker_data:
    print(locker)
