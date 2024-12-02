import math
from typing import Dict, List
from utils.load_data import Location, OrderItem
import random
from multiprocessing import Pool

class Node:
    location: Location
    num_pickup: int
    load: int   
    earliest_time: int
    service_time: int

    def __init__(self):
        self.location = None
        self.num_pickup = 0
        self.load = 0
        self.in_degree = 0
        self.earliest_time = math.inf
        self.service_time = 0

    def set_earliest_time(self, time: int):
        self.earliest_time = min(time, self.earliest_time)

    def set_service_time(self, time: int):
        self.service_time += time

    def __repr__(self) -> str:
        return f"Location={self.location.id}, Load={self.load}, NumPickup={self.num_pickup}, InDegree={self.in_degree}, EarliestTime={self.earliest_time}, ServiceTime={self.service_time}"

class Route:
    def __init__(self):
        self.locations: List[Location] = []

    def get_cost(self) -> float:
        if len(self.locations) < 2:
            return 0
        cost = 0
        for i in range(len(self.locations) - 1):
            current_loc = self.locations[i]
            next_loc = self.locations[i + 1]
            cost += math.sqrt((current_loc.x - next_loc.x) ** 2 + (current_loc.y - next_loc.y) ** 2)
        return cost

    def __len__(self) -> int:
        return len(self.locations)

    def __repr__(self) -> str:
        cost = self.get_cost()
        return f"Cost={cost:.2f}, Locations=[{'->'.join([str(loc.id) for loc in self.locations])}]"

    def add_location(self, location: Location):
        self.locations.append(location)

    def pop_location(self):
        return self.locations.pop()

    def copy(self):
        new_route = Route()
        new_route.locations = self.locations.copy()
        return new_route

class DAG:
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.orders: Dict[int, OrderItem] = {}
        self.adj_matrix: List[List[int]] = []  # adjacency matrix
        self.max_capacity = 100
        self.depot_node = None
        self.id_to_index = {}  # maps location IDs to matrix indices
        self.index_to_id = {}  # maps matrix indices to location IDs
        # Cache the optimal route
        self.changed = False
        self.num_routes = 0
        self.cost = float('inf')
        self.optimal_route = None

    def set_depot(self, depot: Location):
        """Set the depot location"""
        self.depot_node = depot
    
    def _resize_matrix(self, new_id: int):
        """Resize adjacency matrix when adding new nodes"""
        if new_id not in self.id_to_index:
            new_index = len(self.adj_matrix)
            self.id_to_index[new_id] = new_index 
            self.index_to_id[new_index] = new_id
            # Add new row and column
            for row in self.adj_matrix:
                row.append(0)
            self.adj_matrix.append([0] * (new_index + 1))

    def add_order(self, order: OrderItem) -> bool:
        start_id = order.start_location.id
        end_id = order.end_location.id

        if (start_id not in self.nodes) and (end_id not in self.nodes) and self.nodes:
            return False

        # Create nodes if they don't exist
        if start_id not in self.nodes:
            start_node = Node()
            start_node.location = order.start_location
            self.nodes[start_id] = start_node
            self._resize_matrix(start_id)

        if end_id not in self.nodes:
            end_node = Node()
            end_node.location = order.end_location
            self.nodes[end_id] = end_node
            self._resize_matrix(end_id)

        self.changed = True
        # Check for cycles
        if self._check_cycle(start_id, end_id) or self.nodes[start_id].load + order.demand > self.max_capacity:
            return False
        
        # Update pickup count and load
        self.nodes[start_id].num_pickup += 1
        self.nodes[start_id].load += order.demand

        # Add edge to matrix
        start_idx = self.id_to_index[start_id]
        end_idx = self.id_to_index[end_id]
        self.adj_matrix[start_idx][end_idx] = 1  # Changed from += to = 
        self.nodes[end_id].in_degree = sum(self.adj_matrix[i][end_idx] for i in range(len(self.adj_matrix)))  # Fix in_degree calculation
        
        self.orders[order.order_id] = order
        
        # Update the earliest due date
        self.nodes[end_id].set_earliest_time(order.due_date)
        self.nodes[start_id].set_earliest_time(self.nodes[end_id].earliest_time)
        self.nodes[end_id].set_service_time(order.service_time)

        return True

    def _remove_from_matrix(self, node_id: int):
        """Remove a node from the adjacency matrix and update mappings"""
        idx = self.id_to_index[node_id]
        # Remove row
        self.adj_matrix.pop(idx)
        # Remove column from each remaining row
        for row in self.adj_matrix:
            row.pop(idx)
        # Update mappings
        del self.id_to_index[node_id]
        del self.index_to_id[idx]
        # Update indices for nodes after the removed one
        for id, index in self.id_to_index.items():
            if index > idx:
                self.id_to_index[id] = index - 1
                self.index_to_id[index - 1] = id
        del self.index_to_id[len(self.adj_matrix)]

    def remove_order(self, order_id):
        if order_id not in self.orders:
            return
        
        self.changed = True
        order = self.orders[order_id]
        start_id = order.start_location.id
        end_id = order.end_location.id
        
        # Update pickup count and load
        self.nodes[start_id].num_pickup -= 1
        self.nodes[start_id].load -= order.demand
        
        # Remove edge from matrix
        start_idx = self.id_to_index[start_id]
        end_idx = self.id_to_index[end_id]
        self.adj_matrix[start_idx][end_idx] -= 1
        self.nodes[end_id].in_degree -= 1

        # Clean up isolated nodes
        if self.nodes[start_id].in_degree == 0 and self.nodes[start_id].num_pickup == 0:
            self._remove_from_matrix(start_id)
            del self.nodes[start_id]
        if self.nodes[end_id].in_degree == 0:
            self._remove_from_matrix(end_id)
            del self.nodes[end_id]
            
        del self.orders[order_id]

    def _check_cycle(self, start_id: int, end_id: int) -> bool:
        visited = set()
        
        def dfs(node_id: int) -> bool:
            if node_id == start_id:
                return True
            visited.add(node_id)
            curr_idx = self.id_to_index[node_id]
            for next_idx, has_edge in enumerate(self.adj_matrix[curr_idx]):
                if has_edge:
                    next_id = self.index_to_id[next_idx]
                    if next_id not in visited and dfs(next_id):
                        return True
            visited.remove(node_id)
            return False

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
            for node_id, degree in in_degree.items():
                if degree == 0 and node_id not in visited:
                    visited.add(node_id)
                    current_sort.add_location(self.nodes[node_id].location)
                    for next_idx, has_edge in enumerate(self.adj_matrix[self.id_to_index[node_id]]):
                        if has_edge:
                            in_degree[self.index_to_id[next_idx]] -= 1
                    backtrack()
                    visited.remove(node_id)
                    current_sort.pop_location()
                    for next_idx, has_edge in enumerate(self.adj_matrix[self.id_to_index[node_id]]):
                        if has_edge:
                            in_degree[self.index_to_id[next_idx]] += 1

        backtrack()
        if self.depot_node:
            for route in routes:
                route.add_location(self.depot_node)
        return routes

    def valid_routes(self) -> List[Route]:
        """Generate all possible valid topological sorts of the DAG considering capacity constraints"""
        # Recompute in_degrees to ensure correctness
        in_degree = {node_id: sum(self.adj_matrix[i][self.id_to_index[node_id]] 
                                 for i in range(len(self.adj_matrix))) 
                    for node_id in self.nodes}
        
        routes: List[Route] = []
        current_sort = Route()
        visited = set()
        # Track demands that need to be delivered to each location
        pending_demands = {loc_id: 0 for loc_id in self.nodes}
        current_load = 0

        def backtrack():
            nonlocal current_load
            if len(current_sort) == len(self.nodes):
                routes.append(current_sort.copy())
                return

            for node_id, degree in in_degree.items():
                if degree == 0 and node_id not in visited:
                    # Check if adding this node's pickups would exceed capacity
                    new_load = current_load + self.nodes[node_id].load
                    if new_load > self.max_capacity:
                        continue
                        
                    visited.add(node_id)
                    current_sort.add_location(self.nodes[node_id].location)
                    
                    # Update load: add pickups and remove deliveries
                    current_load = new_load - pending_demands[node_id]
                    
                    # Update pending deliveries for destination nodes
                    node_idx = self.id_to_index[node_id]
                    for next_idx, count in enumerate(self.adj_matrix[node_idx]):
                        if count > 0:
                            next_id = self.index_to_id[next_idx]
                            in_degree[next_id] -= 1
                            # Add pending demands for destination
                            for order in self.orders.values():
                                if order.start_location.id == node_id and order.end_location.id == next_id:
                                    pending_demands[next_id] += order.demand

                    backtrack()

                    # Restore state when backtracking
                    visited.remove(node_id)
                    current_sort.pop_location()
                    current_load = new_load - self.nodes[node_id].load
                    
                    node_idx = self.id_to_index[node_id]
                    for next_idx, count in enumerate(self.adj_matrix[node_idx]):
                        if count > 0:
                            next_id = self.index_to_id[next_idx]
                            in_degree[next_id] += 1
                            # Remove pending demands
                            for order in self.orders.values():
                                if order.start_location.id == node_id and order.end_location.id == next_id:
                                    pending_demands[next_id] -= order.demand

        backtrack()
        if self.depot_node:
            for route in routes:
                route.add_location(self.depot_node)
        
        assert len(routes) > 0 and len(self.orders) >= 1, f"""
        Number of routes: {len(routes)}.
        Number of orders: {len(self.orders)}.
        Adjacency matrix: {self.adj_matrix}.
        Nodes: {[node for node in self.nodes.values()]}.
        """
        return routes

    def best_route(self) -> Route:
        """Find the route with shortest cost among all valid routes"""
        valid = self.valid_routes()
        if len(valid) == 0:
            return None
        
        self.num_routes = len(valid)
        min_cost = float('inf')
        best = None
        
        for route in valid:
            cost = route.get_cost()
            if cost < min_cost:
                min_cost = cost
                best = route
        assert best != None, (f"best cannot be None")
        self.cost = min_cost
        self.optimal_route = best        
        return best

    def fitness(self) -> tuple[float, int]:
        """Calculate the fitness of the DAG"""
        assert self.best_route() != None, (f"""
            best_route() must be called before fitness(). 
            Number of orders: {len(self.orders)}.
            {[order for order in self.orders.values()]}
        """)
        assert self.cost < float('inf'), (f"best_route() must be called before fitness(). ")
        return [self.cost, self.num_routes]
    
    def merge_dag(self, other: 'DAG') -> bool:
        cache: 'DAG' = self.copy()
        """Merge another DAG into this one. Returns False if it would create a cycle."""
        # Roll back if it would create a cycle
        for order in other.orders.values():
            if not self.add_order(order):
                self.nodes = cache.nodes
                self.orders = cache.orders
                self.adj_matrix = cache.adj_matrix
                self.id_to_index = cache.id_to_index
                self.index_to_id = cache.index_to_id

                return False
        return True
    
    def copy(self):
        new_dag = DAG()
        new_dag.nodes = self.nodes.copy()
        new_dag.orders = self.orders.copy()
        new_dag.adj_matrix = [row[:] for row in self.adj_matrix]
        new_dag.max_capacity = self.max_capacity
        new_dag.depot_node = self.depot_node
        new_dag.id_to_index = self.id_to_index.copy()
        new_dag.index_to_id = self.index_to_id.copy()
        return new_dag
    
class SASolution:
    def __init__(self):
        self.graphs: List[DAG] = []

    @staticmethod
    def _compute_fitness(graph: DAG) -> float:
        return graph.fitness()[0]
        
    def total_distance(self) -> float:
        with Pool() as pool:
            distances = pool.map(self._compute_fitness, self.graphs)
            return sum(distances)

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

    def random_swap(self) -> bool:
        """Randomly swap two random orders among 2 given DAGs"""
        index_1, index_2 = self.random_pair()
        return self.swap(index_1, index_2)

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
        if len(dag_1.orders) <= 1 and len(dag_2.orders) <= 1:
            return False
        order_1: OrderItem = random.choice(list(dag_1.orders.values()))
        order_2: OrderItem = random.choice(list(dag_2.orders.values()))
        if dag_1.add_order(order_2) and dag_2.add_order(order_1):
            dag_1.remove_order(order_id=order_1.order_id)
            dag_2.remove_order(order_id=order_2.order_id)
            return True
        return False

    def copy(self):
        new_solution = SASolution()
        new_solution.graphs = [dag.copy() for dag in self.graphs]
        return new_solution

def init_solution(orders: List[OrderItem], depot: Location = None, capacity: int = 100) -> SASolution:
    """
    ## Create a tree representation of the route
    """
    solution = SASolution()
    for order in orders:
        dag = DAG()
        dag.max_capacity = capacity
        dag.add_order(order)
        dag.set_depot(depot)
        solution.add_graph(dag)

    return solution