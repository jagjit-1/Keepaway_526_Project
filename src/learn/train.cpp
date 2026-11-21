#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "train.hpp"
#include "approximator.hpp"
#include "sim/keepaway_env.hpp"
#include "tile_coding.hpp"
#include "../apps/common.hpp"

namespace keepaway
{
    Action greedy(const Approximator &q, const std::vector<double> &x)
    {
        int max_action = 0;
        double max_score = 0, curr_val;

        for (int i = 0; i < kNumActions; i++)
        {
            curr_val = q.value(x, possibleActions[i]);

            if (i == 0 || curr_val >= max_score)
            {
                max_score = curr_val;
                max_action = i;
            }
        }

        return possibleActions[max_action];
    }

    double max_over_actions(const Approximator &q, const std::vector<double> &x)
    {
        double max_score = 0, curr_val;
        for (int i = 0; i < kNumActions; i++)
        {
            curr_val = q.value(x, possibleActions[i]);

            if (i == 0 || curr_val >= max_score)
            {
                max_score = curr_val;
            }
        }

        return max_score;
    }

    std::vector<Episode> train_loop(Approximator &q, Env &env, const Config &cfg, const std::vector<Feature> &subset, double gamma, int episodes, uint64_t seed)
    {
        std::vector<Episode> out;
        for (int ep = 0; ep < episodes; ep++)
        {
            StepResult r = env.reset(seed + ep);

            while (!r.done)
            {
                auto x = normalize(select_features(r.next.state, subset), subset, cfg);
                Action a = greedy(q, x);

                int64_t t0 = r.next.t;
                r = env.step(a);
                int64_t k = r.t - t0; // steps until next decision (used for gamma, dicounted reward)
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
}