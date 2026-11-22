#include <fstream>
#include <iomanip>   // setprecision
#include <stdexcept> // runtime_error
#include <cmath>
#include <iostream>

#include "q_learning.hpp"

namespace keepaway
{
    RadialBasisFunction::RadialBasisFunction(const Config_RBF &config)
    {
        num_of_actions_ = config.num_of_actions;
        num_of_state_variables_ = config.num_of_state_variables;
        num_of_tiles_ = config.num_of_tiles;
        standard_deviation_ = config.standard_deviation;
        alpha_ = config.alpha;
        // no tiling concept in rbf, gaussian distribution gives us adjacent value sharing
        weights_ = std::vector<double>(num_of_state_variables_ * num_of_tiles_ * num_of_actions_, 0);
    }

    double RadialBasisFunction::value(const std::vector<double> &x, int action) const
    {
        double total = 0.0;
        const double tile_width = 1.0 / num_of_tiles_;

        for (int sv = 0; sv < num_of_state_variables_; sv++)
        {
            for (int tile = 0; tile < num_of_tiles_; tile++)
            {
                const double mid = (tile_width / 2) + tile * tile_width;
                const double score = calculate_tile_score(x[sv], mid, standard_deviation_);

                total += score * weights_[(sv * num_of_tiles_ * num_of_actions_) + (tile * num_of_actions_) + action];
            }
        }

        return total;
    }

    void RadialBasisFunction::update(const std::vector<double> &x, int action, double target)
    {
        const double tile_width = 1.0 / num_of_tiles_;
        const double error = target - value(x, action);
        const double std_threshold = 3 * standard_deviation_;
        for (int sv = 0; sv < num_of_state_variables_; sv++)
        {
            std::vector<double> scores(num_of_tiles_, 0);
            double squared_score_sum = 0;
            for (int tile = 0; tile < num_of_tiles_; tile++)
            {
                const double mid = (tile_width / 2) + tile * tile_width;
                if (abs(x[sv] - mid) < std_threshold)
                {
                    scores[tile] = calculate_tile_score(x[sv], mid, standard_deviation_);
                    squared_score_sum += scores[tile] * scores[tile];
                }
            }
            const double normalized_aplha = alpha_ / squared_score_sum;
            // std::cout << squared_score_sum << " " << alpha_ << " " << normalized_aplha << " normalized alpha" << '\n';
            for (int tile = 0; tile < num_of_tiles_; tile++)
            {
                // alpha needs to be normalized here by sum of squares of each features
                // reference taken from sutton and barto, p222
                int action_tile = (sv * num_of_tiles_ * num_of_actions_) + (tile * num_of_actions_) + action;
                weights_[action_tile] += normalized_aplha * error * scores[tile];
            }
        }
    }

    double RadialBasisFunction::calculate_tile_score(const double x, const double mean, const double sigma) const
    {
        double z = (x - mean) / sigma;
        return std::exp(-0.5 * z * z); // we dont need the scaling factor here!
    }

    void RadialBasisFunction::save(const std::string &path) const
    {
        std::ofstream out(path);

        if (!out)
            throw std::runtime_error("cannot write to " + path);

        out << num_of_state_variables_ << ' ' << ' ' << num_of_tiles_ << ' ' << num_of_actions_ << '\n';
        out << std::setprecision(17); // exact precision store for weights
        for (double w : weights_)
        {
            out << w << '\n';
        }
    }

    void RadialBasisFunction::load(const std::string &path)
    {
        std::ifstream in(path);

        if (!in)
            throw std::runtime_error("failed to load from" + path);

        int sv, tiles, actions;

        in >> sv >> tiles >> actions;

        if (sv != num_of_state_variables_ || tiles != num_of_tiles_ || actions != num_of_actions_)
        {
            throw std::runtime_error("mismatch in parameters" + path);
        }

        for (double &w : weights_)
        {
            if (!(in >> w))
                throw std::runtime_error("insufficient weight count in" + path);
        }
    }
}
