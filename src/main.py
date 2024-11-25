from utils.load_data import Location, convert_to_orders, load_solomon_vrp
import logging

from utils.route import DAG, Solution, init_routes
logging.basicConfig(level=logging.INFO)

def main():
    # Load the Solomon VRP benchmark file
    vehicle_info, customer_data = load_solomon_vrp('benchmark/C101.txt')
    print("Vehicle Info:", vehicle_info.__dict__)
    # Convert to orders
    orders = convert_to_orders(customer_data)
    print(f"Number of orders: {len(orders)}")
    end_depot = Location(0, customer_data[0].xcoord, customer_data[0].ycoord)
    solution: Solution = init_routes(orders, end_depot)
    # solution.swap(0,1)
    print(f"Num of routes: {len(solution.get_graphs())}")
    solution.random_merge()
    print(f"Num of routes: {len(solution.get_graphs())}")
    for route in solution.get_graphs()[0].all_routes():
        print(route)

if __name__ == "__main__":
    main()
