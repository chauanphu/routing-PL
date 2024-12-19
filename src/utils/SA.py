from utils.route import OrderItem, Location, OrderSet
from typing import List
import random
from multiprocessing import Pool

class SASolution:
    def __init__(self, orders: List[OrderSet] = None):
        self.graphs: List[OrderSet] = orders

    def get_graphs(self) -> List[OrderSet]:
        return self.graphs

    def random_pair(self) -> List[OrderSet]:
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
    
    # def random_decompose(self) -> List[OrderSet]:
    #     """Randomly decompose a OrderSet into smaller DAGs"""
    #     index = random.randint(0, len(self.graphs)-1)
    #     target = self.graphs[index]
        
    #     for order in target.orders.values():
    #         new_dag = OrderSet()
    #         new_dag.set_depot(target.depot_node)
    #         new_dag.add_order(order)
    #         self.graphs.append(new_dag)
            
    #     self.graphs.remove(target)
    #     return self.graphs

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
    
    # def decompose(self, index: int) -> List[OrderSet]:
    #     """Decompose a OrderSet into smaller DAGs"""
    #     target = self.graphs[index]
        
    #     for order in target.orders.values():
    #         new_dag = OrderSet(depot=target.depot_node)
    #         new_dag.add_order(order)
    #         self.graphs.append(new_dag)
            
    #     self.graphs.remove(target)
    #     return self.graphs

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
        dag = OrderSet(depot=depot)
        dag.max_capacity = capacity
        dag.add_order(order)
        solution.add_graph(dag)

    return solution