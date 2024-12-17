import math
from typing import Dict, List
from utils.load_data import Location, OrderItem
import networkx as nx

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
    def __init__(self, depot: Location = None):
        self.locations: List[Location] = []
        self.depot: Location = depot

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

class OrderSet(nx.DiGraph):
    def __init__(self, capacity: int = 0, depot: Location = None):
        super().__init__()
        self.max_capacity = capacity
        self.orders: Dict[int, OrderItem] = {}
        self.depot: Location = depot

    def add_order(self, order: OrderItem):
        self.orders[order.order_id] = order
        # Add weight to the existing start node
        start_loc = order.start_location
        start_id = start_loc.id
        end_loc = order.end_location
        end_id = end_loc.id

        if start_id in self.nodes:
            self.nodes[start_id]['load'] += order.demand
            self.nodes[start_id]['num_pickup'] += 1
            self.nodes[start_id]['service_time'] += order.service_time
            due_time = self.nodes[start_id]['due_time']
            self.nodes[start_id]['due_time'] = min(due_time, order.due_time)
        else:
            self.add_node(
                start_id, 
                load=order.demand,
                num_pickup=1, 
                service_time=order.service_time,
                due_time=order.due_time,
                pos=(start_loc.x, start_loc.y)
            )

        if not end_id in self.nodes:
            self.add_node(
                end_id, 
                load=0,
                num_pickup=0, 
                service_time=0,
                due_time=0,
                pos=(end_loc.x, end_loc.y)
            )
        
        if not self.has_edge(start_id, end_id):
            self.add_edge(start_id, end_id, distance=order.distance)

    def remove_order(self, order: OrderItem):
        self.remove_node(order.start_location.id)
        self.remove_node(order.end_location.id)
        self.orders.pop(order.order_id)
        self.remove_edge(order.start_location.id, order.end_location.id)

    def get_all_routes(self) -> List[Route]:
        sortings = list(nx.all_topological_sorts(self))
        return sortings
    
    def _get_distance(self, u, v):
        if self.has_edge(u, v):
            distance = self.edges[u, v].get('distance', 0)
        else:
            start_loc = self.nodes[u]
            end_loc = self.nodes[v]
            assert start_loc.get('pos') is not None and end_loc.get('pos') is not None, "Location does not have position"
            assert start_loc.get('pos') != end_loc.get('pos'), f"Start and end location are the same: {start_loc.get('pos')}"
            distance = math.sqrt((start_loc['pos'][0] - end_loc['pos'][0]) ** 2 + (start_loc['pos'][1] - end_loc['pos'][1]) ** 2)
        return distance

    def weighted_topological_sort(self, weight='weight'):
        import heapq
        total_distance = 0
        current_weight = 0
        
        # Calculate in-degree for each node
        in_degree = {u: 0 for u in self}
        for u in self:
            for v in self.successors(u):
                in_degree[v] += 1

        # Initialize a heap with nodes of zero in-degree
        heap = []
        for u in self:
            if in_degree[u] == 0:
                w = self.nodes[u].get(weight, 0)
                heapq.heappush(heap, (w, u))

        # Start the route
        route = Route(depot=self.depot)
        current_node = None
        current_load = 0
        current_time = 0

        while heap:
            w, u = heapq.heappop(heap)
            current_location = self.nodes[u]
            current_location = Location(u, current_location['pos'][0], current_location['pos'][1])
            route.add_location(current_location)
            if current_node is not None:
                total_distance += self._get_distance(current_node, u)
                # Update the current weight
                current_weight += w
                current_load += self.nodes[u].get('load', 0)
                current_time += self.nodes[u].get('service_time', 0)
                current_time = max(current_time, self.nodes[u].get('earliest_time', 0))
                current_time += self._get_distance(current_node, u)

            current_node = u
            for v in self.successors(u):
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    w_v = self.nodes[v].get(weight, 0)
                    heapq.heappush(heap, (w_v, v))
        if len(route) != len(self):
            raise nx.NetworkXUnfeasible("Graph contains a cycle.")
        
        return route, route.get_cost()
    
    def fitness(self) -> float:
        """
        Calculate the fitness of the DAG
        """
        total_cost = 0
        return total_cost

# class SASolution:
#     def __init__(self):
#         self.graphs: List[DAG] = []

#     @staticmethod
#     def _compute_fitness(graph: DAG) -> float:
#         return graph.fitness()[0]
        
#     def total_distance(self) -> float:
#         with Pool() as pool:
#             distances = pool.map(self._compute_fitness, self.graphs)
#             return sum(distances)

#     def add_order(self, order: OrderItem):
#         dag = DAG()
#         dag.add_order(order)
#         self.graphs.append(dag)

#     def add_graph(self, graph: DAG):
#         self.graphs.append(graph)

#     def get_graphs(self) -> List[DAG]:
#         return self.graphs

#     def random_pair(self) -> List[DAG]:
#         """Randomly select 2 DAGs"""
#         if len(self.graphs) < 2:
#             return None
#         return random.sample(range(len(self.graphs)), 2)

#     def random_merge(self) -> bool:
#         """Randomly merge two DAGs in the solution. Returns False if it would create a cycle."""
#         [index_1, index_2] = random.sample(range(len(self.graphs)), 2)
#         success = self.graphs[index_1].merge_dag(self.graphs[index_2])
#         if success:
#             self.graphs.pop(index_2)
#         return success
    
#     def random_decompose(self) -> List[DAG]:
#         """Randomly decompose a DAG into smaller DAGs"""
#         index = random.randint(0, len(self.graphs)-1)
#         target = self.graphs[index]
        
#         for order in target.orders.values():
#             new_dag = DAG()
#             new_dag.set_depot(target.depot_node)
#             new_dag.add_order(order)
#             self.graphs.append(new_dag)
            
#         self.graphs.remove(target)
#         return self.graphs

#     def random_swap(self) -> bool:
#         """Randomly swap two random orders among 2 given DAGs"""
#         index_1, index_2 = self.random_pair()
#         return self.swap(index_1, index_2)

#     def merge_dags(self, index_1: int, index_2: int) -> bool:
#         """Merge two DAGs in the solution. Returns False if it would create a cycle."""
#         success = self.graphs[index_1].merge_dag(self.graphs[index_2])
#         if success:
#             self.graphs.pop(index_2)
#         return success
    
#     def decompose(self, index: int) -> List[DAG]:
#         """Decompose a DAG into smaller DAGs"""
#         target = self.graphs[index]
        
#         for order in target.orders.values():
#             new_dag = DAG()
#             new_dag.set_depot(target.depot_node)
#             new_dag.add_order(order)
#             self.graphs.append(new_dag)
            
#         self.graphs.remove(target)
#         return self.graphs
    
#     def swap(self, index_1: int, index_2: int) -> bool:
#         """Swap two random orders among 2 given DAGs"""
#         dag_1 = self.graphs[index_1]
#         dag_2 = self.graphs[index_2]
#         if len(dag_1.orders) <= 1 and len(dag_2.orders) <= 1:
#             return False
#         order_1: OrderItem = random.choice(list(dag_1.orders.values()))
#         order_2: OrderItem = random.choice(list(dag_2.orders.values()))
#         if dag_1.add_order(order_2) and dag_2.add_order(order_1):
#             dag_1.remove_order(order_id=order_1.order_id)
#             dag_2.remove_order(order_id=order_2.order_id)
#             return True
#         return False

#     def copy(self):
#         new_solution = SASolution()
#         new_solution.graphs = [dag.copy() for dag in self.graphs]
#         return new_solution

# def init_solution(orders: List[OrderItem], depot: Location = None, capacity: int = 100) -> SASolution:
#     """
#     ## Create a tree representation of the route
#     """
#     solution = SASolution()
#     for order in orders:
#         dag = DAG()
#         dag.max_capacity = capacity
#         dag.add_order(order)
#         dag.set_depot(depot)
#         solution.add_graph(dag)

#     return solution