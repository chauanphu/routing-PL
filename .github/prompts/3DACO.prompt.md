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

## 3. Route Construction (Greedy insertion heuristics)
- 1. When ant is iteratively traversing the graph, insert new selected customer into the route.
- 2. When visit a customer, the capacity and time-window is updated. The traveling time is the same the distance between nodes.
- 3. If any constraint is violated, switch to next vehicle, back to depot and reset the capacity and time-window.
- 4. If all vehicles are used but there are unserived customers, the solution is infeasible.

We denote each demand by the pair \((i,o)\), where \(i\in N_C\) identifies the customer and \(o\in O\) specifies the chosen option. Routes are constructed incrementally using a Sequential Insertion Heuristic (SIH). Beginning at the depot (\(i=0\)), we select the next demand \((j,o)\) according to the transition probability defined in Equation \ref{eq:3d-transition}. Upon inserting each new demand into the route, we update the vehicle’s load and the current time. This insertion process continues until adding any remaining demand would violate either the vehicle’s capacity or the prescribed time windows. At that point, the route is deemed saturated, and we initiate a new route by resetting both load and time, modeling parallel vehicle deployments.

To streamline bundled deliveries to the same locker, any consecutive demands for that locker are aggregated into a single, larger demand (Figure \ref{fig:vrp-example}). However, revisits to the same locker after servicing an intervening location are treated as violations and hence new vehicle will be deployed. Finally, if unsatisfied demands remain once all vehicles have been deployed, the overall solution is declared infeasible.

# 3D-ACO

Hyperparameters:
- Number of ants $m$
- $\alpha$, $\beta$, $\rho$ and $Q$
- Number of iterations $I$

## Initialization:

1. Construct 3D pheromone matrix: $\tau_{ijo}$, where $i,j \in {N_C + depot}$ and $o \in {0,1}$ delivery options (home, locker).
2. A feasibility mask $M_{ijo} \in \{0, 1\}$ is applied to enforce both home-delivery / locker-delivery constraints. $M_{ij1} = 0, \forall i \in N_C$ if customer $j$ is type-I, and  $M_{ij0} = 0, \forall i \in N_C$ if customer $j$ is type-II. Before computing transition probabilities, we zero out all infeasible entries by masking: $\tilde{\tau}_{ijo} = \tau_{ijo} \times M_{ijo}$. An advantage of feasibility mask is that it only needs to apply once during initialization, since zeroed-out node will have no probability to be visited.

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

Examples in Python

'''python
    
class ThreeDACO(Solver):
    def __init__(self, problem: Problem, num_ants=20, num_iterations=100,
                 alpha=1.0, beta=2.0, evaporation_rate=0.1, Q=1.0):
        """
        Initializes the Sequential ACO optimizer with 3D pheromone.
        """
        super().__init__(problem.node2routes, num_iterations=num_iterations)
        self.problem: Problem = problem
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.Q = Q
        self.tau_min = 0.01 # Min-MAx not implemented
        self.tau_max = 2.0 # Min-max not implemented
        self.num_elitist = 10
        self.n = self.problem.num_customers

        # Initialize 3D pheromone matrix.
        self.pheromones = np.full((self.n + 1, self.n + 1, 2), self.tau_max)
        for j, customer in enumerate(self.problem.customers, start=1):
            if customer.customer_type == 1:
                self.pheromones[:, j, 1] = 0.0
            elif customer.customer_type == 2:
                self.pheromones[:, j, 0] = 0.0

        self.global_best_fitness = float('inf')
        self.best_solution = None  # List of Node objects.
        self.best_solution_transitions = None  # Store transitions for elitist update.
        self.global_best_routes = None
        self.fitness_history = []

    def construct_solution(self) -> tuple:
        """
        Constructs a solution for one ant using 3D pheromone.
        Returns:
            transitions (list[tuple]): List of (prev_index, j, decision) tuples.
            final_route (list[Node]): The constructed route (with depot start and end).
        """
        current_location = (self.problem.depot.x, self.problem.depot.y)
        prev_index = 0  # Depot
        unvisited = list(range(1, self.n + 1))
        transitions = []
        final_route = [self.problem.depot]

        while unvisited:
            candidate_options = []
            for j in unvisited:
                customer = self.problem.customers[j - 1]
                ### Determine allowed delivery modes based on customer type.
                if customer.customer_type == 1:
                    allowed = [0] # Only home delivery allowed.
                elif customer.customer_type == 2:
                    allowed = [1] # Only locker delivery allowed.
                else:
                    allowed = [0, 1] # Both home and locker delivery allowed.
                ### Calculate the value for each candidate option.
                for d in allowed:
                    candidate_coord = (customer.x, customer.y) if d == 0 else (customer.assigned_locker.x, customer.assigned_locker.y)
                    distance = euclidean_distance(current_location, candidate_coord)
                    heuristic = 1.0 / distance if distance > 0 else 1e6
                    tau = self.pheromones[prev_index, j, d] ** self.alpha
                    value = tau * (heuristic ** self.beta)
                    candidate_options.append((j, d, value, candidate_coord)) # Destination, delivery mode, value, coordinates.
            total_value = sum(option[2] for option in candidate_options)
            if total_value > 0:
                probs = [option[2] / total_value for option in candidate_options]
            else:
                probs = [1 / len(candidate_options)] * len(candidate_options)
            r = random.random()
            cumulative = 0.0
            selected = None
            for (j, d, _, candidate_coord), prob in zip(candidate_options, probs):
                cumulative += prob
                if r <= cumulative:
                    selected = (j, d, candidate_coord)
                    break
            if selected is None:
                selected = candidate_options[-1][0:2] + (candidate_options[-1][3],)

            j, d, candidate_coord = selected
            transitions.append((prev_index, j, d))
            node = self.problem.customers[j - 1] if d == 0 else self.problem.customers[j - 1].assigned_locker
            final_route.append(node)
            current_location = candidate_coord
            prev_index = j
            unvisited.remove(j)
        final_route.append(self.problem.depot)
        return transitions, final_route

    def update_pheromones(self, solutions):
        """
        Updates the integrated 3D pheromone matrix based on ant solutions using both max–min and elitist strategies.
        """
        # Evaporate pheromones
        self.pheromones *= (1 - self.evaporation_rate)
        solutions = sorted(solutions, key=lambda x: x[1])
        solutions = solutions[:self.num_elitist]  # Keep only the best solutions

        # Standard deposit from each ant's solution
        for transitions, fitness in solutions:
            deposit = self.Q / (fitness if fitness > 0 else 1e-8)
            for (i, j, d) in transitions:
                self.pheromones[i, j, d] += deposit

        # Max–min pheromone clamping:
        # if self.global_best_fitness < float('inf'):
        #     # Clamp the pheromone matrix to lie within [tau_min, tau_max]
        #     self.pheromones[:] = np.clip(self.pheromones, self.tau_min, self.tau_max)

        # Enforce restrictions on pheromone values for customer types
        for j, customer in enumerate(self.problem.customers, start=1):
            if customer.customer_type == 1:
                self.pheromones[:, j, 1] = 0.0
            elif customer.customer_type == 2:
                self.pheromones[:, j, 0] = 0.0

    def optimize(self, verbose=True):
        """
        Executes the ACO optimization process using 3D pheromone.
        """
        for iteration in range(self.num_iterations):
            solutions = []
            for ant in range(self.num_ants):
                transitions, route = self.construct_solution()
                fitness, routes = self.problem.node2routes(route)
                solutions.append((transitions, fitness))
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.best_solution = route
                    self.best_solution_transitions = transitions
                    self.global_best_routes = routes
            self.update_pheromones(solutions)
            self.fitness_history.append(self.global_best_fitness)
            if verbose:
                print(f"Iteration {iteration + 1}: Best Fitness = {self.global_best_fitness}")
        return self.best_solution, self.global_best_fitness, self.global_best_routes

'''