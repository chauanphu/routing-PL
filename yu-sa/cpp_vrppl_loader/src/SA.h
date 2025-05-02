#pragma once
#include "Solver.h"

class SA : public Solver {
public:
    static Solution solve(const VRPInstance& instance);
};
