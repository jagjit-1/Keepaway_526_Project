#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "learn/approximator.hpp"
#include "sim/keepaway_env.hpp"
#include "../apps/common.hpp"

namespace keepaway
{
    // Action with the highest estimated value in state x.
    Action greedy(const Approximator &q, const std::vector<double> &x);

    // Highest estimated value over all actions in state x.
    double max_over_actions(const Approximator &q, const std::vector<double> &x);

    // Runs `episodes` Q-learning episodes (seeds seed, seed+1, ...)
    std::vector<Episode> train_loop(Approximator &q, Env &env, const Config &cfg, const std::vector<Feature> &subset,
                                    double gamma, int episodes, uint64_t seed);
}
