from typing import Dict, List, Set
from utils.load_data import Location, OrderItem

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
        self.orders: Dict[int, OrderItem] = {}  # location_id -> OrderItem
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
        self.orders[start_id] = order

        return True

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
        result: List[Route] = []
        current_sort = Route()
        visited = set()

        def backtrack():
            if len(current_sort) == len(self.nodes):
                result.append(current_sort.copy())
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
            for route in result:
                route.add_location(self.depot_node)
        return result

    def merge_dag(self, other: 'DAG') -> bool:
        """Merge another DAG into this one. Returns False if it would create a cycle."""
        for node_id, order in other.orders.items():
            if not self.add_order(order):
                return False
        return True

def init_routes(orders: List[OrderItem], depot: Location = None) -> List[DAG]:
    """
    ## Create a tree representation of the route
    """
    dags = []
    for order in orders:
        dag = DAG()
        dag.add_order(order)
        dag.set_depot(depot)
        dags.append(dag)

    return dags