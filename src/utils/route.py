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
            next_distance = ((current_loc.x - next_loc.x) ** 2 + (current_loc.y - next_loc.y) ** 2) ** 0.5
            cost += next_distance
            
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

    def add_order(self, order: OrderItem, priority: float):
        self.orders[order.order_id] = order
        # Add weight to the existing start node
        start_loc = order.start_location
        start_id = start_loc.id
        end_loc = order.end_location
        end_id = end_loc.id

        if start_id in self.nodes:
            self.nodes[start_id]['load'] += order.demand
            self.nodes[start_id]['num_pickup'] += 1
            self.nodes[start_id]['weight'] += priority
            # due_time = self.nodes[start_id]['due_time']
            # self.nodes[start_id]['due_time'] = min(due_time, order.due_time)
        else:
            self.add_node(
                start_id, 
                load=order.demand,
                num_pickup=1, 
                loc=start_loc,
                weight=priority,
            )

        if not end_id in self.nodes:
            self.add_node(
                end_id, 
                load=0,
                num_pickup=0, 
                loc=end_loc,
                weight=0,
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
            start_loc: Location = self.nodes[u]['loc']
            end_loc: Location = self.nodes[v]['loc']
            # Ensure not to calculate the distance of the same node
            assert start_loc != end_loc, f"Start={start_loc}, End={end_loc}"
            distance = ((start_loc.x - end_loc.x) ** 2 + (start_loc.y - end_loc.y) ** 2) ** 0.5
        return distance

    def weighted_topological_sort(self, weight='weight', allow_early=True) -> Route:
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
            current_location: Location = current_location['loc']
            route.add_location(current_location)
            if current_node is not None:
                total_distance += self._get_distance(current_node, u)
                # Update the current weight
                current_weight += w
                current_load += self.nodes[u]['load']
                # Time progress: distance + service time of the current node
                current_time += total_distance + current_location.service_time
                if not allow_early:
                    current_time = max(current_time, current_location.ready_time)
            
            # Apply constraints:
            if current_load > self.max_capacity:
                raise nx.NetworkXUnfeasible("Vehicle capacity exceeded")
            if current_time > current_location.due_time:
                raise nx.NetworkXUnfeasible("Time constraint violated")
            
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

    def isEmpty(self) -> bool:
        return len(self.orders) == 0