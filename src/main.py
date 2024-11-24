from typing import List
from utils.load_data import Location, convert_to_orders, load_solomon_vrp
import logging

from utils.route import DAG, init_routes
logging.basicConfig(level=logging.INFO)

def main():
    # Load the Solomon VRP benchmark file
    vehicle_info, customer_data = load_solomon_vrp('benchmark/C101.txt')
    print("Vehicle Info:", vehicle_info.__dict__)
    # Convert to orders
    orders = convert_to_orders(customer_data)
    print(f"Number of orders: {len(orders)}")
    end_depot = Location(0, customer_data[0].xcoord, customer_data[0].ycoord)
    trees: List[DAG] = init_routes(orders, end_depot)
    trees[0].merge_dag(trees[1])
    topo_sorts = trees[0].all_routes()
    for route in topo_sorts:
        print(route)

if __name__ == "__main__":
    main()
