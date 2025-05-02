#include "Solver.h"
#include <limits>

Solution Solver::solve(const VRPInstance& instance) {
    Solution sol;
    // Pseudo-solver: always return infeasible
    sol.routes.clear();
    sol.objective_value = std::numeric_limits<double>::max();
    return sol;
}
