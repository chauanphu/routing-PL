Based on the instance description in `VRPPL.prompt.md`, start to implement the VRPPL solver

# Pseudo-solver interface:
- Input: an instance
- Output: Best solution (routes), best objective value. If infeasible, then best solutin is empty and objective value is abitrary high.

## 1. Objectie value
- The objective is the total traveling distance between nodes
- Calculated using Euclidean distance.
- To save computation, the distance matrix is calculated during instance parsing.
- Infeasible solution will receive very high penalty.

## 2. Constraints
- Capacity: total carrying demand must not exceed vehicle capacity
- Time-window: early node visit will render the vehicle to wait, however, late visit is not permitted.

## 3. Route Construction
- 1. Based on the delivery node corresponding to a each customer, a vehicle traverses the graph, starting from depot 0's location and back to depot.
- 2. When visit a customer, the capacity and time-window is updated. The traveling time is the same the distance between nodes.
- 3. If any constraint is violated, switch to next vehicle, back to depot and reset the capacity and time-window.
- 4. If all vehicles are used but there are unserived customers, the solution is infeasible.

# Simulated Annealing

Hyperparameters: Maximum number of iteration $I_{iter}$, Initial temperature $T_0$ and maximum non-improving iter $N_{patience}$, $\alpha$ and $\beta$

## Initial solution:

Solution Representation as 2 array:
1. First array: permutation of customer, sorted based on their indices.
2. Second array: assigned particular delivery node based on type of customer. Type-I will be assigned directly to their node, type-II to their prefered locker node, type-III be assigned randomly based on $p$
3. Consecutive delivery nodes to the same locker will be aggregated into a single, larger demand. Revisit the locker after serving is treated as violation. For example: delivery nodes [27, 27, 27 ,8] is allowed but [27, 27, 8, 27] is violated.

For example:
```
First array: 1, 2, 3, 4, 5,...N
Second array: 1, 27, 3, 26, 26,...
```

Next, nearest neighborhood is performed, selects closest node according to the current node. Performed until all customers have been assigned.

