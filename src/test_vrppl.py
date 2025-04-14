import unittest
import math
import random

from meta.solver import Customer, Node, Problem

# Assuming the classes (Problem, Node, Customer, Vehicle) have been defined or imported
# from the module where the VRPPL code resides.

# For testing purposes, we assume that the classes are available in the current context.
# (If they are in another module, e.g., `vrppl`, then you could do: 
#    from vrppl import Problem, Node, Customer, Vehicle)

class TestVRPPL(unittest.TestCase):

    def setUp(self):
        # For reproducibility when randomness is involved (e.g., locker assignment for type 3)
        random.seed(42)

    def test_position2route_total_distance(self):
        """
        Create a simple instance with two type 1 customers.
        Expect the route: depot -> customer1 -> customer2 -> depot
        and the total distance = 1 (0->(1,0)) + 1 ((1,0)->(2,0)) + 2 ((2,0)->0) = 4.
        """
        problem = Problem()
        # Define depot at (0,0) with wide time window.
        problem.depot = Node(node_id=0, x=0.0, y=0.0, early=0.0, late=100.0, service=0.0)
        # Two customers (type 1: always home delivery).
        cust1 = Customer(node_id=1, x=1.0, y=0.0, early=0.0, late=100.0, service=0.0, demand=1, customer_type=1)
        cust2 = Customer(node_id=2, x=2.0, y=0.0, early=0.0, late=100.0, service=0.0, demand=1, customer_type=1)
        problem.customers = [cust1, cust2]
        problem.lockers = []  # No lockers needed
        problem.num_customers = 2
        problem.num_lockers = 0
        problem.num_vehicles = 1
        problem.vehicle_capacity = 3

        # Provide positions that force the order [cust1, cust2]
        positions = [0.1, 0.2]
        total_distance, routes = problem.position2route(positions)
        
        expected_distance = 1 + 1 + 2  # 1: depot->cust1, 1: cust1->cust2, 2: cust2->depot
        self.assertAlmostEqual(total_distance, expected_distance)

        # Check that the returned route is as expected: depot, cust1, cust2, depot.
        route_ids = [node.node_id for node in routes[0]]
        self.assertEqual(route_ids, [0, 1, 2, 0])

    def test_time_window_constraint(self):
        """
        Create an instance where the customer’s time window makes service impossible.
        The customer is located far enough that even after waiting, the service would start after the allowed window.
        The decoder should return infeasibility (total_distance = float('inf') and no routes).
        """
        problem = Problem()
        problem.depot = Node(node_id=0, x=0.0, y=0.0, early=0.0, late=100.0, service=0.0)
        # Place customer at (10,0) but with a very narrow time window [2,3]
        cust = Customer(node_id=1, x=10.0, y=0.0, early=2.0, late=3.0, service=0.0, demand=1, customer_type=1)
        problem.customers = [cust]
        problem.lockers = []
        problem.num_customers = 1
        problem.num_lockers = 0
        problem.num_vehicles = 1
        problem.vehicle_capacity = 10

        positions = [0.5]
        total_distance, routes = problem.position2route(positions)
        self.assertEqual(total_distance, float('inf'))
        self.assertEqual(routes, [])

    def test_capacity_constraint(self):
        """
        Create an instance where the customer's demand exceeds the vehicle capacity.
        The solution decoder should detect the capacity violation and return infeasible.
        """
        problem = Problem()
        problem.depot = Node(node_id=0, x=0.0, y=0.0, early=0.0, late=100.0, service=0.0)
        # Customer demand is 10 while capacity is only 5.
        cust = Customer(node_id=1, x=1.0, y=0.0, early=0.0, late=100.0, service=0.0, demand=10, customer_type=1)
        problem.customers = [cust]
        problem.lockers = []
        problem.num_customers = 1
        problem.num_lockers = 0
        problem.num_vehicles = 1
        problem.vehicle_capacity = 5

        positions = [0.5]
        total_distance, routes = problem.position2route(positions)
        self.assertEqual(total_distance, float('inf'))
        self.assertEqual(routes, [])

    def test_duplicate_locker_violation(self):
        """
        Create an instance where two type 2 customers share the same assigned locker,
        but the order forces a non-consecutive locker visit (i.e. depot -> locker -> other node -> locker),
        which should trigger the duplicate locker constraint.
        """
        problem = Problem()
        # Define depot.
        problem.depot = Node(node_id=0, x=0.0, y=0.0, early=0.0, late=100.0, service=0.0)
        # Create a locker.
        locker = Node(node_id=3, x=10.0, y=0.0, early=0.0, late=100.0, service=0.0)
        problem.lockers = [locker]
        # Customer 1: type 2 (always locker delivery), assigned to locker.
        cust1 = Customer(node_id=1, x=8.0, y=0.0, early=0.0, late=100.0, service=0.0,
                         demand=1, customer_type=2, assigned_locker=locker)
        # Customer 2: type 1 (home delivery).
        cust2 = Customer(node_id=2, x=1.0, y=0.0, early=0.0, late=100.0, service=0.0,
                         demand=1, customer_type=1)
        # Customer 3: type 2 (locker delivery), assigned to the same locker.
        cust3 = Customer(node_id=4, x=9.0, y=0.0, early=0.0, late=100.0, service=0.0,
                         demand=1, customer_type=2, assigned_locker=locker)

        problem.customers = [cust1, cust2, cust3]
        problem.num_customers = 3
        problem.num_lockers = 1
        # Only one vehicle available so that the duplicate locker violation cannot be “reset” by starting a new route.
        problem.num_vehicles = 1
        problem.vehicle_capacity = 10

        # Force the ordering: [cust1, cust2, cust3]
        positions = [0.1, 0.2, 0.3]
        total_distance, routes = problem.position2route(positions)
        # Expect infeasibility due to the duplicate locker violation.
        self.assertEqual(total_distance, float('inf'))
        self.assertEqual(routes, [])

    def test_permu2route_correctness(self):
        """
        Test the route decoding from a permutation (first and last elements are depots).
        For two type 1 customers, the expected route is: depot -> customer1 -> customer2 -> depot.
        """
        problem = Problem()
        depot = Node(node_id=0, x=0.0, y=0.0, early=0.0, late=100.0, service=0.0)
        problem.depot = depot
        cust1 = Customer(node_id=1, x=1.0, y=0.0, early=0.0, late=100.0, service=0.0,
                         demand=1, customer_type=1)
        cust2 = Customer(node_id=2, x=2.0, y=0.0, early=0.0, late=100.0, service=0.0,
                         demand=1, customer_type=1)
        problem.customers = [cust1, cust2]
        problem.lockers = []
        problem.num_customers = 2
        problem.num_lockers = 0
        problem.num_vehicles = 1
        problem.vehicle_capacity = 3

        # Permutation: [depot, cust1, cust2, depot]
        permutation = [0, 1, 2, 0]
        total_distance, routes = problem.permu2route(permutation)
        expected_distance = 1 + 1 + 2  # 4.0
        self.assertAlmostEqual(total_distance, expected_distance)
        route_ids = [node.node_id for node in routes[0]]
        self.assertEqual(route_ids, [0, 1, 2, 0])

    def test_node2routes_correctness(self):
        """
        Test node2routes which decodes a permutation of Node objects (including the depot) 
        into a route. For two type 1 customers, the expected route is: depot -> customer1 -> customer2 -> depot.
        """
        problem = Problem()
        depot = Node(node_id=0, x=0.0, y=0.0, early=0.0, late=100.0, service=0.0)
        problem.depot = depot
        cust1 = Customer(node_id=1, x=1.0, y=0.0, early=0.0, late=100.0, service=0.0,
                         demand=1, customer_type=1)
        cust2 = Customer(node_id=2, x=2.0, y=0.0, early=0.0, late=100.0, service=0.0,
                         demand=1, customer_type=1)
        problem.customers = [cust1, cust2]
        # For this test, we include a dummy locker (won't be used).
        locker = Node(node_id=3, x=3.0, y=0.0, early=0.0, late=100.0, service=0.0)
        problem.lockers = [locker]
        problem.num_customers = 2
        problem.num_lockers = 1
        problem.num_vehicles = 1
        problem.vehicle_capacity = 3

        # Build a permutation of Node objects: [depot, cust1, cust2, depot]
        permutation = [depot, cust1, cust2, depot]
        total_distance, routes = problem.node2routes(permutation)
        expected_distance = 1 + 1 + 2  # same as before: 4.0
        self.assertAlmostEqual(total_distance, expected_distance)
        route_ids = [node.node_id for node in routes[0]]
        self.assertEqual(route_ids, [0, 1, 2, 0])

    def test_multi_route_solution_position2route(self):
        """
        Create a scenario where the vehicle capacity forces the solution to be split into multiple routes.
        We set vehicle_capacity=1 and each customer has a demand of 1.
        For three customers located at (1,0), (2,0), and (3,0) respectively:
        Expected routes:
            Route1: depot -> customer1 -> depot  with distance: 1 + 1 = 2
            Route2: depot -> customer2 -> depot  with distance: 2 + 2 = 4
            Route3: depot -> customer3 -> depot  with distance: 3 + 3 = 6
        Total distance expected = 2 + 4 + 6 = 12.
        """
        problem = Problem()
        depot = Node(node_id=0, x=0.0, y=0.0, early=0.0, late=100.0, service=0.0)
        problem.depot = depot
        cust1 = Customer(node_id=1, x=1.0, y=0.0, early=0.0, late=100.0, service=0.0,
                         demand=1, customer_type=1)
        cust2 = Customer(node_id=2, x=2.0, y=0.0, early=0.0, late=100.0, service=0.0,
                         demand=1, customer_type=1)
        cust3 = Customer(node_id=3, x=3.0, y=0.0, early=0.0, late=100.0, service=0.0,
                         demand=1, customer_type=1)
        problem.customers = [cust1, cust2, cust3]
        problem.lockers = []
        problem.num_customers = 3
        problem.num_lockers = 0
        # We provide 3 vehicles to allow one customer per route.
        problem.num_vehicles = 3
        problem.vehicle_capacity = 1

        # Provide positions in an order that forces the algorithm to process them sequentially.
        positions = [0.1, 0.2, 0.3]
        total_distance, routes = problem.position2route(positions)
        
        expected_total = 2 + 4 + 6  # 12
        self.assertAlmostEqual(total_distance, expected_total)
        # Check that we indeed have 3 routes.
        self.assertEqual(len(routes), 3)
        # Verify each route starts and ends with the depot.
        for route in routes:
            self.assertEqual(route[0].node_id, 0)
            self.assertEqual(route[-1].node_id, 0)

    def test_multi_route_solution_permu2route(self):
        """
        Similar to test_multi_route_solution_position2route but using the permutation decoding method.
        We force a scenario where capacity forces the use of multiple routes.
        """
        problem = Problem()
        depot = Node(node_id=0, x=0.0, y=0.0, early=0.0, late=100.0, service=0.0)
        problem.depot = depot
        # Three customers, each with demand=1, placed at increasing distances.
        cust1 = Customer(node_id=1, x=1.0, y=0.0, early=0.0, late=100.0, service=0.0,
                         demand=1, customer_type=1)
        cust2 = Customer(node_id=2, x=2.0, y=0.0, early=0.0, late=100.0, service=0.0,
                         demand=1, customer_type=1)
        cust3 = Customer(node_id=3, x=3.0, y=0.0, early=0.0, late=100.0, service=0.0,
                         demand=1, customer_type=1)
        problem.customers = [cust1, cust2, cust3]
        problem.lockers = []
        problem.num_customers = 3
        problem.num_lockers = 0
        # Provide 3 vehicles (one per customer) with capacity=1.
        problem.num_vehicles = 3
        problem.vehicle_capacity = 1

        # Permutation: [depot, cust1, cust2, cust3, depot]
        permutation = [0, 1, 2, 3, 0]
        total_distance, routes = problem.permu2route(permutation)
        # Expected distances:
        # Route1: depot->cust1->depot = 1+1=2, Route2: depot->cust2->depot = 2+2=4, Route3: depot->cust3->depot = 3+3=6.
        expected_total = 2 + 4 + 6  # 12
        self.assertAlmostEqual(total_distance, expected_total)
        self.assertEqual(len(routes), 3)
        for route in routes:
            self.assertEqual(route[0].node_id, 0)
            self.assertEqual(route[-1].node_id, 0)

    def test_multi_route_solution_node2routes(self):
        """
        Test the multi-route case using node2routes.
        We use three customers with demand=1 and capacity=1 to force three routes.
        """
        problem = Problem()
        depot = Node(node_id=0, x=0.0, y=0.0, early=0.0, late=100.0, service=0.0)
        problem.depot = depot
        cust1 = Customer(node_id=1, x=1.0, y=0.0, early=0.0, late=100.0, service=0.0,
                         demand=1, customer_type=1)
        cust2 = Customer(node_id=2, x=2.0, y=0.0, early=0.0, late=100.0, service=0.0,
                         demand=1, customer_type=1)
        cust3 = Customer(node_id=3, x=3.0, y=0.0, early=0.0, late=100.0, service=0.0,
                         demand=1, customer_type=1)
        problem.customers = [cust1, cust2, cust3]
        # Include a dummy locker.
        locker = Node(node_id=4, x=4.0, y=0.0, early=0.0, late=100.0, service=0.0)
        problem.lockers = [locker]
        problem.num_customers = 3
        problem.num_lockers = 1
        problem.num_vehicles = 3
        problem.vehicle_capacity = 1

        # Build a permutation of Node objects: [depot, cust1, cust2, cust3, depot]
        permutation = [depot, cust1, cust2, cust3, depot]
        total_distance, routes = problem.node2routes(permutation)
        expected_total = 2 + 4 + 6  # 12
        self.assertAlmostEqual(total_distance, expected_total)
        self.assertEqual(len(routes), 3)
        for route in routes:
            self.assertEqual(route[0].node_id, 0)
            self.assertEqual(route[-1].node_id, 0)
            
if __name__ == '__main__':
    unittest.main()