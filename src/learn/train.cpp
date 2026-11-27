// ---------------------------------------------------------------------------
// train.cpp
// Reinforcement learning training loops for Keepaway 3-v-2.
//
// Implements:
//   - greedy and epsilon-greedy action selection
//   - train_loop (Q-learning or Sarsa, chosen by RLAlgorithm, with epsilon-greedy exploration)
//   - evaluate_policy (evaluation of learned policy without updates)
//
// SMDP Temporal-Difference target formulation:
//   Terminal:    target = -gamma^(k - 1)
//   Nonterminal:
//     Sarsa:      target = gamma^k * Q(next_state, next_action)
//     Q-learning: target = gamma^k * max_a Q(next_state, a)
//
// Partially generated with assistance from AI (DeepMind Antigravity / Claude)
// and adapted/verified by human authors.
//
// Changed by an LLM: Claude (Anthropic), 2026-11-23, at Jagjit's request:
// train_rl now applies the update to the state where the action was taken
// (q.update(x, a, target) before x = x2; previously x was overwritten first).
//
// Changed by an LLM: Claude (Anthropic), 2026-11-26, at Jagjit's request:
// train_rl was merged into train_loop, which now takes an RLAlgorithm. Q-learning
// behaves exactly as the previous train_loop and Sarsa exactly as the previous
// train_rl Sarsa branch (checked: identical results for the same seeds).
// train_rl was removed.
// ---------------------------------------------------------------------------

#include "train.hpp"
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <random>

#include "approximator.hpp"
#include "sim/keepaway_env.hpp"
#include "q_learning.hpp"
#include "../apps/common.hpp"

namespace keepaway {

Action greedy(const Approximator& q, const std::vector<double>& x) {
  int max_action = 0;
  double max_score = -1e9;

  for (int i = 0; i < kNumActions; ++i) {
    double curr_val = q.value(x, possibleActions[i]);
    if (i == 0 || curr_val > max_score) {
      max_score = curr_val;
      max_action = i;
    }
  }

  return possibleActions[max_action];
}

Action epsilon_greedy(const Approximator& q, const std::vector<double>& x, double epsilon, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  if (dist(rng) < epsilon) {
    std::uniform_int_distribution<int> act_dist(0, kNumActions - 1);
    return possibleActions[act_dist(rng)];
  }
  return greedy(q, x);
}

double max_over_actions(const Approximator& q, const std::vector<double>& x) {
  double max_score = -1e9;
  for (int i = 0; i < kNumActions; ++i) {
    double curr_val = q.value(x, possibleActions[i]);
    if (i == 0 || curr_val > max_score) {
      max_score = curr_val;
    }
  }
  return max_score;
}

std::vector<Episode> train_loop(Approximator& q, Env& env, const Config& cfg, const std::vector<Feature>& subset,
                                RLAlgorithm algo, double gamma, int episodes, uint64_t seed, double epsilon) {
  // One SMDP TD(0) loop for both algorithms; the behaviour policy is epsilon-greedy in both.
  //   Q-learning: target uses max_a Q(next_state, a); the next action is chosen after the update.
  //   Sarsa:      the next action is chosen (epsilon-greedily) before the update, used in the
  //               target, and then played.
  std::vector<Episode> out;
  out.reserve(episodes);
  std::mt19937_64 rng(seed);

  for (int ep = 0; ep < episodes; ++ep) {
    StepResult r = env.reset(seed + ep);
    std::vector<double> x;
    Action a{};
    if (!r.done) {
      x = normalize(select_features(r.next.state, subset), subset, cfg);
      a = epsilon_greedy(q, x, epsilon, rng);
    }

    while (!r.done) {
      int64_t t0 = r.next.t;
      r = env.step(a);
      int64_t k = r.t - t0;  // number of simulator steps elapsed for this decision

      if (r.done) {
        q.update(x, a, -std::pow(gamma, k - 1));
        break;
      }

      auto x2 = normalize(select_features(r.next.state, subset), subset, cfg);
      Action next_a;
      if (algo == RLAlgorithm::Sarsa) {
        next_a = epsilon_greedy(q, x2, epsilon, rng);
        q.update(x, a, std::pow(gamma, k) * q.value(x2, next_a));
      } else {
        q.update(x, a, std::pow(gamma, k) * max_over_actions(q, x2));
        next_a = epsilon_greedy(q, x2, epsilon, rng);
      }
      x = std::move(x2);
      a = next_a;
    }
    out.push_back({env.episode_seconds(), env.end_reason()});
  }

  return out;
}

std::vector<Episode> evaluate_policy(const Approximator& q, Env& env, const Config& cfg,
                                     const std::vector<Feature>& subset, int episodes, uint64_t seed) {
  std::vector<Episode> out;
  out.reserve(episodes);

  for (int ep = 0; ep < episodes; ++ep) {
    StepResult r = env.reset(seed + ep);

    while (!r.done) {
      auto x = normalize(select_features(r.next.state, subset), subset, cfg);
      Action a = greedy(q, x);
      r = env.step(a);
    }
    out.push_back({env.episode_seconds(), env.end_reason()});
  }

  return out;
}

}  // namespace keepaway
