Based on the instance description in `VRPPL.prompt.md`, start to implement the VRPPL solver

# Pseudo-solver interface:
- Input: an instance
- Output: Best solution (routes), best objective value. If infeasible, then best solutin is empty and objective value is abitrary high.

# Simulated Annealing

Hyperparameters: Maximum number of iteration $I_{iter}$, Initial temperature $T_0$ and maximum non-improving iter $N_{patience}$, $\alpha$ and $\beta$

## Initial solution:

Solution Representation as 2 array:
1. First array: permutation of customer, sorted based on their indices.
2. Second array: assigned particular delivery node based on type of customer. Type-I will be assigned directly to their node, type-II to their prefered locker node, type-III be assigned randomly based on $p$

For example:
```
First array: 1, 2, 3, 4, 5,...N
Second array: 1, 27, 3, 26, 26,...
```

Next, nearest neighborhood is performed, selects closest node according to the current node. Performed until all customers have been assigned.

