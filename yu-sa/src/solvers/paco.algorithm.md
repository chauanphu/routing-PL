# PACO::solve Algorithmic Description

\begin{algorithm}[H]
\caption{3D-PACO for VRPPL with Adaptive GA Elite Refinement}
\label{alg:aco-vrppl}
\begin{algorithmic}
\STATE
\STATE \textbf{Input:} Graph $G=(N,E)$
\STATE \textbf{Parameters:} $\alpha,\beta, \rho, Q, m_{min}$ (ants per thread), $p$ (processors), $t$ (top elite ants), $I_{max}$ (max stagnation), $I_{LS}$ (local search iters)
\STATE \textbf{Output:} Best solution $S^*$ and objective value $D^*$
\STATE
\STATE Initialize pheromone matrix $\tau_{ijo}$
\STATE Construct infeasibility mask $M_{ijo}$
\STATE $S^* \gets \emptyset$, $D^* \gets \infty$, $I_{stag} \gets 0$
\WHILE{$I_{stag} \le I_{max}$}
    \STATE $S_{\text{iter}} \gets \emptyset$
    \STATE $m_p \gets m_{min} \cdot \min(I_{stag}+1, 8)$
    \FOR{each processor $j = 1$ to $p$ \textbf{in parallel}}
        \STATE $S^{(j)}_{\text{local}} \gets \emptyset$
        \FOR{each ant $a = 1$ to $m_p$}
            \STATE Construct solution $S_a$ using 3D-ACO transition rule
            \FOR{$I_{LS}$ local search iterations}
                \STATE Apply local search (swap, insert, 2-opt) to $S_a$
            \ENDFOR
            \STATE Evaluate $D_{S_a}$
            \STATE Add $S_a$ to $S^{(j)}_{\text{local}}$
        \ENDFOR
        % --- Adaptive GA for Elite Refinement ---
        \STATE Select $t_j$ best solutions from $S^{(j)}_{\text{local}}$ as GA population
        \STATE Set $G_{\text{GA}} \gets$ base generations $+$ bonus $\propto I_{stag}/I_{max}$
        \FOR{generation $g = 1$ to $G_{\text{GA}}$}
            \FOR{offspring $o = 1$ to $t_j$}
                \STATE Select parents by tournament selection
                \STATE Apply order crossover (OX) to generate child
                \STATE Apply mutation (swap) with probability $p_{mut}$
                \STATE Evaluate child and add to offspring pool
            \ENDFOR
            \STATE Combine parents and offspring, select best $t_j$ for next generation (elitism)
        \ENDFOR
        \STATE Replace $S^{(j)}_{\text{local}}$ with final GA population
    \ENDFOR
    \STATE Wait for all $p$ processors to finish
    \STATE $S_{\text{iter}} \gets \bigcup_{j=1}^{p} S^{(j)}_{\text{local}}$
    \STATE Sort $S_{\text{iter}}$ by $D$
    \STATE Select top $t$ elite solutions from $S_{\text{iter}}$
    \STATE Update $\tau_{ijo}$ using selected solutions
    \STATE Let $S_{\text{best}}$ be the best in $S_{\text{iter}}$, $D_{\text{best}}$ its cost
    \IF{$D_{\text{best}} < D^*$}
        \STATE $S^* \gets S_{\text{best}},\; D^* \gets D_{\text{best}}$
        \STATE $I_{stag} \gets 0$
    \ELSE
        \STATE $I_{stag} \gets I_{stag} + 1$
    \ENDIF
    \STATE Adapt $\rho$ if needed (e.g., sigmoid schedule)
\ENDWHILE
\STATE \textbf{return} $S^*, D^*$
\end{algorithmic}
\end{algorithm}

---

## Overview
`PACO` is a hybrid metaheuristic for solving a variant of the Vehicle Routing Problem (VRP) using a Parallel Ant Colony Optimization (PACO) approach, enhanced with local search and a genetic algorithm (GA) for elite solution refinement. The method is parallelized using OpenMP and supports multi-threaded execution.

---

## Step-by-Step Algorithm

### 1. Initialization
- **Parameters**: Extracts problem and algorithm parameters (number of ants `m`, threads `p`, elite solutions `t`, max non-improved iterations `I`, etc.).
- **Pheromone Matrix**: Initializes a 3D pheromone matrix `tau` and a feasibility mask `mask` to encode allowed transitions and delivery types (direct or locker).
- **Global Best**: Sets up variables to track the global best solution and convergence history.

### 2. Main Iterative Loop
The main loop continues until the number of non-improved iterations reaches `I`.

#### 2.1. Parallel Ant Construction and Local Search
- **Parallelization**: The loop is parallelized over `p` threads.
- **Ant Construction**: Each thread constructs multiple ant solutions using a probabilistic transition rule based on pheromone and heuristic information.
- **Local Search**: Each ant solution undergoes local search (swap, insert, reverse/2-opt) to improve its permutation.
- **Evaluation**: Each solution is evaluated for objective value.

#### 2.2. Genetic Algorithm (GA) for Elite Solutions
- **GA Population**: Each thread selects its best ant solutions to form a local GA population.
- **Tournament Selection**: Parents are selected using tournament selection.
- **Crossover & Mutation**: Offspring are generated using order crossover and swap mutation.
- **Replacement**: Generational replacement with elitism is used to keep the best solutions.
- **GA Generations**: The number of generations is adaptive, increasing with the number of non-improved iterations.

#### 2.3. Global Solution Pool
- **Aggregation**: Each thread contributes its best solutions (from GA or fallback to best ants) to a global pool for the current iteration.
- **Sorting**: The global pool is sorted by objective value.

#### 2.4. Pheromone Update
- **Evaporation**: All pheromone values are decayed by a factor `(1 - rho)`.
- **Reinforcement**: The top `t` solutions from the global pool reinforce the pheromone matrix along their routes, proportional to their quality.

#### 2.5. Global Best Update & Convergence
- **Best Solution**: If a new best solution is found, it is recorded and the non-improved counter is reset.
- **Evaporation Rate**: The evaporation rate `rho` is adaptively updated using a sigmoid function based on the number of non-improved iterations.
- **Convergence History**: The best objective value is recorded for each iteration if history tracking is enabled.

### 3. Output
- **Convergence Log**: If enabled, writes the convergence history to a CSV file.
- **Return**: Returns the best solution found.

---

## Key Features
- **Parallelism**: Uses OpenMP for multi-threaded ant construction and GA operations.
- **Hybridization**: Combines ACO with local search and a genetic algorithm for elite solution refinement.
- **Adaptive Parameters**: Adjusts the number of ants and GA generations based on search progress.
- **Flexible Delivery Types**: Supports both direct and locker deliveries with feasibility masking.

---

## References
- PACO: Population-based Ant Colony Optimization
- VRP: Vehicle Routing Problem
- OX: Order Crossover (GA)
- 2-opt: Local search for permutations

---

*This document was generated by analyzing the implementation in `PACO.cpp`.*
