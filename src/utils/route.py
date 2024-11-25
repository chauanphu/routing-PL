from typing import Dict, List, Set
from utils.load_data import Location, OrderItem
import random

class Node:
    location: Location
    num_pickup: int
    load: int   
    
    def __init__(self):
        self.location = None
        self.num_pickup = 0
        self.load = 0
        self.in_degree = 0

class Route:
    def __init__(self):
        self.cost = 0
        self.locations: List[Location] = []

    def __len__(self) -> int:
        return len(self.locations)

    def __repr__(self) -> str:
        return f"Cost={self.cost:.2f}, Locations=[{'->'.join([str(loc.id) for loc in self.locations])}]"

    def add_location(self, location: Location):
        if len(self.locations) > 0:
            last_location = self.locations[-1]
            self.cost += ((last_location.x - location.x) ** 2 + (last_location.y - location.y) ** 2) ** 0.5
        self.locations.append(location)

    def copy(self):
        new_route = Route()
        new_route.cost = self.cost
        new_route.locations = self.locations.copy()
        return new_route

class DAG:
    """A DAG representation of the order sets"""
    def __init__(self):
        self.nodes: Dict[int, Node] = {}  # location_id -> Node
        self.orders: Dict[int, OrderItem] = {}  # order_id -> OrderItem
        self.edges: Dict[int, Set[int]] = {}  # location_id -> set of child location_ids
        self.max_capacity = 100
        self.depot_node = None

    def set_depot(self, depot: Location):
        """Set the depot location"""
        self.depot_node = depot

    def add_order(self, order: OrderItem) -> bool:
        """Add a new order to the DAG. Returns False if it would create a cycle."""
        start_id = order.start_location.id
        end_id = order.end_location.id
        # If this is seperate order from the graph
        if (start_id not in self.nodes) and (end_id not in self.nodes) and self.nodes:
            return False

        # Create nodes if they don't exist
        if start_id not in self.nodes:
            start_node = Node()
            start_node.location = order.start_location
            self.nodes[start_id] = start_node
            self.edges[start_id] = set()

        if end_id not in self.nodes:
            end_node = Node()
            end_node.location = order.end_location
            self.nodes[end_id] = end_node
            self.edges[end_id] = set()
        # Check if adding this edge would create a cycle
        if self._would_create_cycle(start_id, end_id):
            return False
        
        # Update pickup count and load
        self.nodes[start_id].num_pickup += 1
        self.nodes[start_id].load += order.demand

        # Add edge
        self.nodes[end_id].in_degree += 1
        self.edges[start_id].add(end_id)
        self.orders[order.order_id] = order

        return True

    def remove_order(self, order_id):
        """Remove an order from the DAG"""
        if order_id not in self.orders:
            return
        order = self.orders[order_id]
        start_id = order.start_location.id
        end_id = order.end_location.id
        # Update pickup count and load
        self.nodes[start_id].num_pickup -= 1
        self.nodes[start_id].load -= order.demand
        # Remove edge
        self.nodes[end_id].in_degree -= 1
        self.edges[start_id].remove(end_id)
        # Remove start and end nodes if they are not connected to any other nodes
        if self.nodes[start_id].in_degree == 0 and self.nodes[start_id].num_pickup == 0:
            del self.nodes[start_id]
            del self.edges[start_id]
        if self.nodes[end_id].in_degree == 0:
            del self.nodes[end_id]
            del self.edges[end_id]
        del self.orders[order_id]

    def _would_create_cycle(self, start_id: int, end_id: int) -> bool:
        """Check if adding an edge would create a cycle using DFS"""
        visited = set()
        
        def dfs(node_id: int) -> bool:
            if node_id == start_id:
                return True
            visited.add(node_id)
            for next_id in self.edges[node_id]:
                if next_id not in visited and dfs(next_id):
                    return True
            visited.remove(node_id)
            return False

        # First check if end_id can reach start_id
        return dfs(end_id)

    def all_routes(self) -> List[Route]:
        """Generate all possible topological sorts of the DAG"""
        # Copy in-degrees so we don't modify the original graph
        in_degree = {node_id: self.nodes[node_id].in_degree for node_id in self.nodes}
        routes: List[Route] = []
        current_sort = Route()
        visited = set()
        def backtrack():
            if len(current_sort) == len(self.nodes):
                routes.append(current_sort.copy())
                return

            # Find all nodes with in-degree 0 that haven't been used
            for node_id in self.nodes:
                if in_degree[node_id] == 0 and node_id not in visited:
                    # Add current node to the sort
                    visited.add(node_id)
                    location = self.nodes[node_id].location
                    current_sort.add_location(location)

                    # Reduce in-degree for all neighbors
                    for neighbor in self.edges[node_id]:
                        in_degree[neighbor] -= 1
                    backtrack()
                    # Backtrack: restore state
                    visited.remove(node_id)
                    current_sort.locations.pop()
                    for neighbor in self.edges[node_id]:
                        in_degree[neighbor] += 1

        backtrack()
        if self.depot_node:
            for route in routes:
                route.add_location(self.depot_node)
        return routes

    # Similar to all_routes, but it only returns valid routes (i.e. routes that don't exceed the capacity)
    def valid_routes(self) -> List[Route]:
        pass

    def merge_dag(self, other: 'DAG') -> bool:
        cache = self.copy()
        """Merge another DAG into this one. Returns False if it would create a cycle."""
        # Roll back if it would create a cycle
        for order in other.orders.values():
            if not self.add_order(order):
                self.nodes = cache.nodes
                self.orders = cache.orders
                self.edges = cache.edges
                return False
        return True
    
    def copy(self):
        new_dag = DAG()
        new_dag.nodes = self.nodes.copy()
        new_dag.orders = self.orders.copy()
        new_dag.edges = self.edges.copy()
        new_dag.max_capacity = self.max_capacity
        new_dag.depot_node = self.depot_node
        return new_dag

class Solution:
    def __init__(self):
        self.graphs: List[DAG] = []
        
    def add_order(self, order: OrderItem):
        dag = DAG()
        dag.add_order(order)
        self.graphs.append(dag)

    def add_graph(self, graph: DAG):
        self.graphs.append(graph)

    def get_graphs(self) -> List[DAG]:
        return self.graphs

    def random_pair(self) -> List[DAG]:
        """Randomly select 2 DAGs"""
        if len(self.graphs) < 2:
            return None
        return random.sample(range(len(self.graphs)), 2)

    def random_merge(self) -> bool:
        """Randomly merge two DAGs in the solution. Returns False if it would create a cycle."""
        [index_1, index_2] = random.sample(range(len(self.graphs)), 2)
        success = self.graphs[index_1].merge_dag(self.graphs[index_2])
        if success:
            self.graphs.pop(index_2)
        return success
    
    def random_decompose(self) -> List[DAG]:
        """Randomly decompose a DAG into smaller DAGs"""
        index = random.randint(0, len(self.graphs)-1)
        target = self.graphs[index]
        
        for order in target.orders.values():
            new_dag = DAG()
            new_dag.set_depot(target.depot_node)
            new_dag.add_order(order)
            self.graphs.append(new_dag)
            
        self.graphs.remove(target)
        return self.graphs

    def merge_dags(self, index_1: int, index_2: int) -> bool:
        """Merge two DAGs in the solution. Returns False if it would create a cycle."""
        success = self.graphs[index_1].merge_dag(self.graphs[index_2])
        if success:
            self.graphs.pop(index_2)
        return success
    
    def decompose(self, index: int) -> List[DAG]:
        """Decompose a DAG into smaller DAGs"""
        target = self.graphs[index]
        
        for order in target.orders.values():
            new_dag = DAG()
            new_dag.set_depot(target.depot_node)
            new_dag.add_order(order)
            self.graphs.append(new_dag)
            
        self.graphs.remove(target)
        return self.graphs
    
    def swap(self, index_1: int, index_2: int) -> bool:
        """Swap two random orders among 2 given DAGs"""
        dag_1 = self.graphs[index_1]
        dag_2 = self.graphs[index_2]
        order_1: OrderItem = random.choice(list(dag_1.orders.values()))
        order_2: OrderItem = random.choice(list(dag_2.orders.values()))
        if dag_1.add_order(order_2) and dag_2.add_order(order_1):
            dag_1.remove_order(order_id=order_1.order_id)
            dag_2.remove_order(order_id=order_2.order_id)
            return True
        return False

    def random_swap(self) -> bool:
        """Randomly swap two orders among 2 random DAGs"""
        [index_1, index_2] = self.random_pair()
        return self.swap(index_1, index_2)

def init_routes(orders: List[OrderItem], depot: Location = None) -> Solution:
    """
    ## Create a tree representation of the route
    """
    solution = Solution()
    for order in orders:
        dag = DAG()
        dag.add_order(order)
        dag.set_depot(depot)
        solution.add_graph(dag)

    return solution