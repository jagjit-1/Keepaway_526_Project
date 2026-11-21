#include "tile_coding.hpp"
#include <vector>
#include <fstream>   // ofstream / ifstream: write to / read from a file
#include <iomanip>   // setprecision
#include <stdexcept> // runtime_error

using namespace std;

// Implementation inspiration taken from

namespace keepaway
{

    TileCoding::TileCoding(const Config_TC &config)
    {
        num_of_tiles_ = config.num_of_tiles;
        num_of_actions_ = config.num_of_actions;
        num_of_tilings_ = config.num_of_tilings;
        alpha_ = config.alpha;
        num_of_state_variables_ = config.num_of_state_variables;
        // using hyperplane slicing and not regular grid, num_of_tiles + 1, because of shifting offsets
        weights_ = vector<double>(num_of_state_variables_ * num_of_tilings_ * (num_of_tiles_ + 1) * num_of_actions_, 0);
        // TODO: think about any combinations of state variables we should also add?
    };

    double TileCoding::value(const vector<double> &x, int action) const
    {
        // get tiles which are active and add them up to get expected
        double val = 0.0;
        for (int i : active_tiles(x, action))
        {
            val += weights_[i];
        }
        return val;
    }

    void TileCoding::update(const vector<double> &x, int action, double target)
    {
        // q_i+1 <- q_i + alpha(error)
        double error = target - value(x, action);
        for (int i : active_tiles(x, action))
        {
            weights_[i] += alpha_ * error;
        }
    }

    vector<int> TileCoding::active_tiles(const vector<double> &x, int action) const
    {
        vector<int> result;
        for (int sv = 0; sv < num_of_state_variables_; sv++)
        {
            for (int tiling = 0; tiling < num_of_tilings_; tiling++)
            {
                // since every feature value is normalized, the width of each tiling is 1
                double tile_width = 1.0 / num_of_tiles_;
                double tile_shift = (tiling * tile_width) / num_of_tilings_;

                for (int tile = 0; tile < num_of_tiles_ + 1; tile++)
                {
                    if (x[sv] >= tile * tile_width - tile_shift && x[sv] < (tile + 1) * tile_width - tile_shift)
                    {
                        // with action block
                        result.push_back(
                            sv * (num_of_tilings_ * (num_of_tiles_ + 1) * num_of_actions_) +
                            tiling * ((num_of_tiles_ + 1) * num_of_actions_) +
                            tile * num_of_actions_ +
                            action);
                    }
                }
            }
        }

        return result;
    }

    void TileCoding::save(const string &path) const
    {
        ofstream out(path);

        if (!out)
            throw runtime_error("cannot write to" + path);

        out << num_of_state_variables_ << ' ' << num_of_tilings_ << ' ' << num_of_tiles_ << ' ' << num_of_actions_ << '\n';
        out << setprecision(17); // exact precision store for weights
        for (double w : weights_)
        {
            out << w << '\n';
        }
    }

    void TileCoding::load(const string &path)
    {
        ifstream in(path);

        if (!in)
            throw runtime_error("failed to load from" + path);

        int sv, tilings, tiles, actions;

        in >> sv >> tilings >> tiles >> actions;

        if (sv != num_of_state_variables_ || tilings != num_of_tilings_ || tiles != num_of_tiles_ || actions != num_of_actions_)
        {
            throw runtime_error("mismatch in parameters" + path);
        }

        for (double &w : weights_)
        {
            if (!(in >> w))
                throw runtime_error("insufficient weight count in" + path);
        }
    }
}