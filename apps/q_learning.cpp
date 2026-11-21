#include <filesystem>
#include <iostream>

#include "sim/keepaway_env.hpp"
#include "learn/tile_coding.hpp"
#include "learn/train.hpp"
#include "common.hpp"

namespace keepaway
{
    Row train_tilecoding_full(int episodes, const std::string &out_dir)
    {
        // default : one learning keeper, two hardcoded
        Config cfg;
        Env env(cfg);

        const auto &subset = kFullFeatures;
        Config_TC config;
        // Check these config values, we might need to do some tests to arrive at optimal configs
        config.num_of_actions = kNumActions;
        config.num_of_tiles = 10;
        config.num_of_tilings = 10;
        config.num_of_state_variables = subset.size();
        config.alpha = 0.1 / (config.num_of_state_variables * config.num_of_tilings);

        TileCoding q(config);
        std::vector<Row> rows;
        rows.push_back({"TileCoding", train_loop(q, env, cfg, subset, 0.9, episodes, 1)});
        save_csv(rows, out_dir + "/tc_full_seed1.csv");

        return rows[0];
    }
}

int main(int argc, char **argv)
{
    std::filesystem::create_directories("results");
    std::string out_dir = "results";
    int episodes = 1000;

    if (argc > 1)
    {
        episodes = std::atoi(argv[1]);
    }
    std::vector<Row> rows;

    rows.push_back(keepaway::train_tilecoding_full(episodes, out_dir));

    for (const Row &r : rows)
        print_row(r.name, r.eps);
}