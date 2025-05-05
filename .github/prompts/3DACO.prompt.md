Based on the instance description in `VRPPL.prompt.md`, start to implement the VRPPL solver, called `PACO.h / PACO.cpp`

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

## 3. Route Construction (Greedy insertion heuristics)
- 1. When ant is iteratively traversing the graph, insert new selected customer into the route.
- 2. When visit a customer, the capacity and time-window is updated. The traveling time is the same the distance between nodes.
- 3. If any constraint is violated, switch to next vehicle, back to depot and reset the capacity and time-window.
- 4. If all vehicles are used but there are unserived customers, the solution is infeasible.

## Delivery options assignment:
- We denote each demand by the pair \((i,o)\), where \(i\in N_C\) identifies the customer and \(o\in O\) specifies the chosen option.
- Routes are constructed incrementally using a Sequential Insertion Heuristic (SIH).
 - Beginning at the depot (\(i=0\)), we select the next demand \((j,o)\) according to the transition probability defined in Equation \ref{eq:3d-transition}.
 - Upon inserting each new demand into the route, we update the vehicle’s load and the current time.
 - This insertion process continues until adding any remaining demand would violate either the vehicle’s capacity or the prescribed time windows. 
 - At that point, the route is deemed saturated, and we initiate a new route by resetting both load and time, modeling parallel vehicle deployments.

- To streamline bundled deliveries to the same locker, any consecutive demands for that locker are aggregated into a single, larger demand (Figure \ref{fig:vrp-example}). However, revisits to the same locker after servicing an intervening location are treated as violations and hence new vehicle will be deployed. Finally, if unsatisfied demands remain once all vehicles have been deployed, the overall solution is declared infeasible.

# 3D-ACO

## Hyperparameters:
- Number of ants $m$
- $\alpha$, $\beta$, $\rho$ and $Q$
- Number of iterations $I$
- $t$: top ants for pheromone update (elitist strategy)
- $p$: number of processors required

## Initialization:

1. Construct 3D pheromone matrix: $\tau_{ijo}$, where $i,j \in {N_C + depot}$ and $o \in {0,1}$ delivery options (home, locker).
2. A feasibility mask $M_{ijo} \in \{0, 1\}$ is applied to enforce both home-delivery / locker-delivery constraints.
 - $M_{ij1} = 0, \forall i \in N_C$ if customer $j$ is type-I
 - $M_{ij0} = 0, \forall i \in N_C$ if customer $j$ is type-II.
 - $M_{iio} = 0 \forall i \in N_C$ to prevent self-delivery.
 - Before computing transition probabilities, we zero out all infeasible entries by masking: $\tilde{\tau}_{ijo} = \tau_{ijo} \times M_{ijo}$.
  - An advantage of feasibility mask is that it only needs to apply **once during initialization**, since zeroed-out node will have no probability to be visited.

## Iteration
1. Similar to traditional ACO but with 3D pheromone
2. Applied top-t elitist strategy so that only best $t$ ants is allowed to update the matrix.
\begin{equation}
\Delta\tau^{(k)}_{ijo} =
\begin{cases}
\displaystyle\frac{Q}{L^{(k)}}, & \text{if } (i,j,o) \in \text{solution path of } k \\
0, & \text{otherwise}
\end{cases}
\end{equation}
\begin{equation}
\tau_{ijo} \leftarrow (1 - \rho) \cdot \tau_{ijo} + \sum_{k \in t} \Delta\tau^{(k)}_{ijo}
\label{3d-phero-update}
\end{equation}

3. Transitional probability:
\begin{equation}
\Delta\tau^{(k)}_{ijo} =
\begin{cases}
\displaystyle\frac{Q}{L^{(k)}}, & \text{if } (i,j,o) \in \text{solution path of } k \\
0, & \text{otherwise}
\end{cases}

\end{equation}
\begin{equation}
\tau_{ijo} \leftarrow (1 - \rho) \cdot \tau_{ijo} + \sum_{k \in t} \Delta\tau^{(k)}_{ijo}
\label{3d-phero-update}
\end{equation}

## Parallelization
It uses coarse-grain master-slave parallel schema.

### Master:
- Manages global resources: shared-memory 3d pheromone matrix, best known solution / objectives
- Master will handle the pheromone update for each iteration, only select top-$k$ ants (solutions) to do it.

### Slaves:
- Each slave is a parallel processes, each slave will have a subcolony of ants: $m_p = m // p$, m is the total ants, p is the number of processors 
- From the shared-memory 3d pheromone matrix, each slave constructs solutions.
- In each iteration, wait until all slaves have completed, then update the pheromone and global best known solution (if found)
- Start again for new iteration, until completed.