#pragma once
#include "Solver.h"

struct SAParams {
    int max_iter = 1000;
    double T0 = 10.0;
    double Tf = 0.1;
    int patience = 50;
    double alpha = 0.97;
    double beta = 1.0;
    double p = 0.5; // for type-III locker assignment
};

class SA : public Solver {
public:
    static Solution solve(const VRPInstance& instance);
    static Solution solve(const VRPInstance& instance, const SAParams& params);
private:
    static Solution iterate(const VRPInstance& instance, std::vector<int> customer_perm, std::unordered_map<int, int> customer2node, const SAParams& params);
};
