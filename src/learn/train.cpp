// ---------------------------------------------------------------------------
// train.cpp
// Implementation of greedy/epsilon-greedy policies and SMDP TD training loops.
// ---------------------------------------------------------------------------

#include "train.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "approximator.hpp"
#include "sim/keepaway_env.hpp"
#include "tile_coding.hpp"
#include "../apps/common.hpp"

namespace keepaway
{
    Action greedy(const Approximator &q, const std::vector<double> &x)
    {
        int max_action = 0;
        double max_score = -1e30;

        for (int i = 0; i < kNumActions; i++)
        {
            double curr_val = q.value(x, possibleActions[i]);
            if (i == 0 || curr_val > max_score)
            {
                max_score = curr_val;
                max_action = i;
            }
        }

        return possibleActions[max_action];
    }

    Action epsilon_greedy(const Approximator &q, const std::vector<double> &x,
                          double epsilon, std::mt19937_64 &rng)
    {
        if (epsilon > 0.0)
        {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            if (dist(rng) < epsilon)
            {
                std::uniform_int_distribution<int> act_dist(0, kNumActions - 1);
                return possibleActions[act_dist(rng)];
            }
        }
        return greedy(q, x);
    }

    double max_over_actions(const Approximator &q, const std::vector<double> &x)
    {
        double max_score = -1e30;
        for (int i = 0; i < kNumActions; i++)
        {
            double curr_val = q.value(x, possibleActions[i]);
            if (i == 0 || curr_val > max_score)
            {
                max_score = curr_val;
            }
        }

        return max_score;
    }

    std::vector<Episode> train_loop(Approximator &q, Env &env, const Config &cfg,
                                    const std::vector<Feature> &subset, double gamma,
                                    int episodes, uint64_t seed, double epsilon)
    {
        std::mt19937_64 rng(seed * 6364136223846793005ULL + 1442695040888963407ULL);
        std::vector<Episode> out;
        out.reserve(episodes);

        for (int ep = 0; ep < episodes; ep++)
        {
            StepResult r = env.reset(seed + ep);

            while (!r.done)
            {
                auto x = normalize(select_features(r.next.state, subset), subset, cfg);
                Action a = (epsilon > 0.0) ? epsilon_greedy(q, x, epsilon, rng) : greedy(q, x);

                int64_t t0 = r.next.t;
                r = env.step(a);
                int64_t k = r.t - t0; // steps until next decision (discounted reward)
                double target;

                if (r.done)
                {
                    target = -std::pow(gamma, k - 1);
                }
                else
                {
                    auto x2 = normalize(select_features(r.next.state, subset), subset, cfg);
                    target = std::pow(gamma, k) * max_over_actions(q, x2);
                }

                q.update(x, a, target);
            }
            out.push_back({env.episode_seconds(), env.end_reason()});
        }

        return out;
    }

    std::vector<Episode> train_sarsa_loop(Approximator &q, Env &env, const Config &cfg,
                                          const std::vector<Feature> &subset, double gamma,
                                          int episodes, uint64_t seed, double epsilon)
    {
        std::mt19937_64 rng(seed * 6364136223846793005ULL + 1442695040888963407ULL);
        std::vector<Episode> out;
        out.reserve(episodes);

        for (int ep = 0; ep < episodes; ep++)
        {
            StepResult r = env.reset(seed + ep);
            if (r.done)
            {
                out.push_back({env.episode_seconds(), env.end_reason()});
                continue;
            }

            auto x = normalize(select_features(r.next.state, subset), subset, cfg);
            Action a = (epsilon > 0.0) ? epsilon_greedy(q, x, epsilon, rng) : greedy(q, x);

            while (!r.done)
            {
                int64_t t0 = r.next.t;
                r = env.step(a);
                int64_t k = r.t - t0;

                double target;
                Action a2 = a;
                std::vector<double> x2;

                if (r.done)
                {
                    target = -std::pow(gamma, k - 1);
                }
                else
                {
                    x2 = normalize(select_features(r.next.state, subset), subset, cfg);
                    a2 = (epsilon > 0.0) ? epsilon_greedy(q, x2, epsilon, rng) : greedy(q, x2);
                    target = std::pow(gamma, k) * q.value(x2, a2);
                }

                q.update(x, a, target);

                if (!r.done)
                {
                    x = std::move(x2);
                    a = a2;
                }
            }
            out.push_back({env.episode_seconds(), env.end_reason()});
        }

        return out;
    }

    std::vector<Episode> evaluate_policy(const Approximator &q, Env &env, const Config &cfg,
                                         const std::vector<Feature> &subset,
                                         int episodes, uint64_t seed)
    {
        std::vector<Episode> out;
        out.reserve(episodes);

        for (int ep = 0; ep < episodes; ep++)
        {
            StepResult r = env.reset(seed + ep);
            while (!r.done)
            {
                auto x = normalize(select_features(r.next.state, subset), subset, cfg);
                Action a = greedy(q, x);
                r = env.step(a);
            }
            out.push_back({env.episode_seconds(), env.end_reason()});
        }

        return out;
    }

} // namespace keepaway
