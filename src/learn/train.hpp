// ---------------------------------------------------------------------------
// train.hpp
// Training and evaluation routines for Keepaway value-function approximators.
// ---------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "learn/approximator.hpp"
#include "sim/keepaway_env.hpp"
#include "../apps/common.hpp"

namespace keepaway
{
    // Action with the highest estimated value in state x.
    Action greedy(const Approximator &q, const std::vector<double> &x);

    // Epsilon-greedy action selection.
    Action epsilon_greedy(const Approximator &q, const std::vector<double> &x,
                          double epsilon, std::mt19937_64 &rng);

    // Highest estimated value over all actions in state x.
    double max_over_actions(const Approximator &q, const std::vector<double> &x);

    // Runs `episodes` SMDP Q-learning episodes.
    // Default epsilon = 0.0 maintains backwards compatibility.
    std::vector<Episode> train_loop(Approximator &q, Env &env, const Config &cfg,
                                    const std::vector<Feature> &subset, double gamma,
                                    int episodes, uint64_t seed, double epsilon = 0.0);

    // Runs `episodes` SMDP Sarsa episodes (on-policy TD, as in Stone et al., 2005).
    std::vector<Episode> train_sarsa_loop(Approximator &q, Env &env, const Config &cfg,
                                          const std::vector<Feature> &subset, double gamma,
                                          int episodes, uint64_t seed, double epsilon = 0.01);

    // Evaluates a trained policy (using greedy action selection) without updating weights.
    std::vector<Episode> evaluate_policy(const Approximator &q, Env &env, const Config &cfg,
                                         const std::vector<Feature> &subset,
                                         int episodes, uint64_t seed);
}
