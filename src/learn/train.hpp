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
//
// Changed by an LLM: Claude (Anthropic), 2026-11-26, at Jagjit's request: train_loop
// takes an RLAlgorithm (Q-learning or Sarsa) and has no default epsilon; train_rl was
// removed.
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

// SMDP TD(0) training loop with epsilon-greedy exploration.
//   algo = QLearning: target = gamma^k * max_a Q(next_state, a)
//   algo = Sarsa:     target = gamma^k * Q(next_state, next_action), next_action chosen epsilon-greedily
//   Terminal decisions (both): target = -gamma^(k - 1)
std::vector<Episode> train_loop(Approximator& q, Env& env, const Config& cfg, const std::vector<Feature>& subset,
                                RLAlgorithm algo, double gamma, int episodes, uint64_t seed, double epsilon);

// Greedy evaluation loop: runs episodes without exploration or weight updates
std::vector<Episode> evaluate_policy(const Approximator& q, Env& env, const Config& cfg,
                                     const std::vector<Feature>& subset, int episodes, uint64_t seed);

}  // namespace keepaway
