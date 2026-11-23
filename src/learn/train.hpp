// ---------------------------------------------------------------------------
// train.hpp
// Training and evaluation loops for reinforcement learning in 3-v-2 Keepaway.
//
// Supports:
//   - Action selection: greedy and epsilon-greedy
//   - Algorithm modes: Sarsa (on-policy) and Q-learning (off-policy)
//   - Training loop with SMDP temporal-difference targets and epsilon-greedy exploration
//   - Evaluation loop (greedy policy execution without exploration)
//
// Partially generated with assistance from AI (DeepMind Antigravity / Claude)
// and adapted/verified by human authors.
// ---------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <random>

#include "learn/approximator.hpp"
#include "sim/keepaway_env.hpp"
#include "../apps/common.hpp"

namespace keepaway {

enum class RLAlgorithm {
  Sarsa,
  QLearning
};

// Returns action with the highest estimated Q-value in state x.
Action greedy(const Approximator& q, const std::vector<double>& x);

// Epsilon-greedy action selection: picks random action with probability epsilon, else greedy.
Action epsilon_greedy(const Approximator& q, const std::vector<double>& x, double epsilon, std::mt19937_64& rng);

// Highest estimated value over all actions in state x.
double max_over_actions(const Approximator& q, const std::vector<double>& x);

// Q-learning train loop with epsilon-greedy exploration (default epsilon = 0.05)
std::vector<Episode> train_loop(Approximator& q, Env& env, const Config& cfg, const std::vector<Feature>& subset,
                                double gamma, int episodes, uint64_t seed, double epsilon = 0.05);

// Generalized RL train loop supporting Sarsa & Q-learning with epsilon-greedy exploration
std::vector<Episode> train_rl(Approximator& q, Env& env, const Config& cfg, const std::vector<Feature>& subset,
                              RLAlgorithm algo, double gamma, double epsilon, int episodes, uint64_t seed);

// Greedy evaluation loop: runs episodes without exploration or weight updates
std::vector<Episode> evaluate_policy(const Approximator& q, Env& env, const Config& cfg,
                                     const std::vector<Feature>& subset, int episodes, uint64_t seed);

}  // namespace keepaway
