#pragma once
#include "Solver.h"

struct GAParams {
    int population_size = 50;
    int generations = 1000;
    double crossover_rate = 0.8;
    double mutation_rate = 0.2;
    double p = 0.5; // for type-III locker assignment
};

class GA : public Solver {
public:
    static Solution solve(const VRPInstance& instance);
    static Solution solve(const VRPInstance& instance, const GAParams& params);
private:
    static Solution iterate(const VRPInstance& instance, std::vector<int> customer_perm, std::unordered_map<int, int> customer2node, const GAParams& params);
};
