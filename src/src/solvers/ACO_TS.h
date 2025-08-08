#pragma once
#include "../core/VRPInstance.h"
#include "Solver.h"
#include <vector>
#include <unordered_map>
#include <random>

struct ACOTSParams {
    int num_ants = 50;
    int num_iterations = 1000;
    double alpha = 1.0;
    double beta = 2.0;
    double rho = 0.1; // evaporation
    double Q = 1.0;
    int stagnation_limit = 10;
    double p = 0.5; // for type-III assignment
};

class ACO_TS {
public:
    static Solution solve(const VRPInstance& instance, const ACOTSParams& params, bool history = false);
};
