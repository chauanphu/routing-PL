#pragma once
#include "VRPInstance.h"
#include <vector>

struct Solution {
    std::vector<std::vector<int>> routes; // Each route is a sequence of node IDs
    double objective_value;
};

class Solver {
public:
    static Solution solve(const VRPInstance& instance);
};
