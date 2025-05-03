#pragma once
#include "../core/VRPInstance.h"
#include "Solver.h"
#include <vector>
#include <random>
#include <tuple>

struct ThreeDACOParams {
    int num_ants = 20;
    int num_iterations = 100;
    double alpha = 1.0;
    double beta = 2.0;
    double evaporation_rate = 0.1;
    double Q = 1.0;
    int num_elitist = 10;
};

class ThreeDACO {
public:
    static Solution solve(const VRPInstance& instance, const ThreeDACOParams& params);
};
