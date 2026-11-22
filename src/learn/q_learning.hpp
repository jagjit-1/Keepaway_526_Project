#pragma once

#include "approximator.hpp"
#include <vector>

namespace keepaway
{
    struct Config_TC
    {
        int num_of_state_variables;
        int num_of_tilings;
        int num_of_tiles;
        int num_of_actions;
        double alpha;
    };

    struct Config_RBF
    {
        int num_of_state_variables;
        int num_of_tiles;
        int num_of_actions;
        double standard_deviation;
        double alpha;
    };

    class TileCoding : public Approximator
    {
    public:
        // tile coding properties
        // no of tiles per tilling, no of tiling per state variable
        // learning rate alpha
        TileCoding(const Config_TC &config);

        double value(const std::vector<double> &x, int action) const override;
        void update(const std::vector<double> &x, int action, double target) override;
        void save(const std::string &path) const override;
        void load(const std::string &path) override;

    private:
        std::vector<int> active_tiles(const std::vector<double> &x, int action) const;
        int num_of_tiles_, num_of_tilings_, num_of_actions_, num_of_state_variables_;
        double alpha_;
        std::vector<double> weights_;
    };

    class RadialBasisFunction : public Approximator
    {
    public:
        // tile coding properties
        // no of tiles per tilling, no of tiling per state variable
        // learning rate alpha
        RadialBasisFunction(const Config_RBF &config);

        double value(const std::vector<double> &x, int action) const override;
        void update(const std::vector<double> &x, int action, double target) override;
        void save(const std::string &path) const override;
        void load(const std::string &path) override;

    private:
        double calculate_tile_score(const double x, const double mean, const double sigma) const;
        int num_of_tiles_, num_of_actions_, num_of_state_variables_;
        double alpha_, standard_deviation_;
        std::vector<double> weights_;
    };
}