// ---------------------------------------------------------------------------
// neural_net.cpp
// Implementation of feedforward neural network function approximators
// for Keepaway soccer reinforcement learning.
// ---------------------------------------------------------------------------

#include "neural_net.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>

namespace keepaway
{

    namespace
    {
        inline double sigmoid(double z)
        {
            // Clamped to avoid floating point under/overflow
            double cz = std::clamp(z, -45.0, 45.0);
            return 1.0 / (1.0 + std::exp(-cz));
        }
    } // namespace

    // ---------------------------------------------------------------------------
    // FeedForwardNet (Sub-network for a single action, 13-20-1)
    // ---------------------------------------------------------------------------

    FeedForwardNet::FeedForwardNet(int num_in, int num_hidden, double alpha, double momentum,
                                   double weight_decay, double gradient_clip, uint64_t seed)
        : num_in_(num_in),
          num_hidden_(num_hidden),
          alpha_(alpha),
          momentum_(momentum),
          weight_decay_(weight_decay),
          gradient_clip_(gradient_clip)
    {
        w1_.resize(num_hidden_ * num_in_);
        b1_.resize(num_hidden_, 0.0);
        w2_.resize(num_hidden_);
        b2_ = 0.0;

        v_w1_.assign(w1_.size(), 0.0);
        v_b1_.assign(b1_.size(), 0.0);
        v_w2_.assign(w2_.size(), 0.0);
        v_b2_ = 0.0;

        init_weights(seed);
    }

    void FeedForwardNet::init_weights(uint64_t seed)
    {
        std::mt19937_64 rng(seed);

        // Xavier / Glorot uniform initialization for sigmoid hidden layer
        double limit1 = std::sqrt(6.0 / (num_in_ + num_hidden_));
        std::uniform_real_distribution<double> dist1(-limit1, limit1);
        for (double &w : w1_)
        {
            w = dist1(rng);
        }
        for (double &b : b1_)
        {
            b = 0.0;
        }

        // Output layer (linear) initialization
        double limit2 = std::sqrt(6.0 / (num_hidden_ + 1));
        std::uniform_real_distribution<double> dist2(-limit2, limit2);
        for (double &w : w2_)
        {
            w = dist2(rng);
        }
        b2_ = 0.0;
    }

    double FeedForwardNet::predict(const std::vector<double> &x) const
    {
        if (static_cast<int>(x.size()) != num_in_)
        {
            throw std::runtime_error("FeedForwardNet input dimension mismatch: expected " +
                                     std::to_string(num_in_) + ", got " + std::to_string(x.size()));
        }

        // Hidden layer activations
        double out = b2_;
        for (int j = 0; j < num_hidden_; ++j)
        {
            double z = b1_[j];
            const int offset = j * num_in_;
            for (int i = 0; i < num_in_; ++i)
            {
                z += w1_[offset + i] * x[i];
            }
            double h = sigmoid(z);
            out += w2_[j] * h;
        }

        return out;
    }

    void FeedForwardNet::train_step(const std::vector<double> &x, double target)
    {
        if (static_cast<int>(x.size()) != num_in_)
        {
            throw std::runtime_error("FeedForwardNet input dimension mismatch");
        }

        // 1. Forward pass, caching hidden activations
        std::vector<double> h(num_hidden_);
        double y_hat = b2_;

        for (int j = 0; j < num_hidden_; ++j)
        {
            double z = b1_[j];
            const int offset = j * num_in_;
            for (int i = 0; i < num_in_; ++i)
            {
                z += w1_[offset + i] * x[i];
            }
            h[j] = sigmoid(z);
            y_hat += w2_[j] * h[j];
        }

        // 2. TD Error delta = target - y_hat
        double delta = target - y_hat;
        if (gradient_clip_ > 0.0)
        {
            delta = std::clamp(delta, -gradient_clip_, gradient_clip_);
        }

        // 3. Backprop to output layer
        // Loss E = 1/2 * (y_hat - target)^2 => -dE/dy_hat = delta
        // Output weights: -dE/dw2_j = delta * h_j
        // Output bias: -dE/db2 = delta
        std::vector<double> old_w2 = w2_; // preserve for backprop into hidden layer

        for (int j = 0; j < num_hidden_; ++j)
        {
            double grad_w2 = delta * h[j] - weight_decay_ * w2_[j];
            v_w2_[j] = momentum_ * v_w2_[j] + alpha_ * grad_w2;
            w2_[j] += v_w2_[j];
        }
        v_b2_ = momentum_ * v_b2_ + alpha_ * delta;
        b2_ += v_b2_;

        // 4. Backprop to hidden layer
        // delta_h[j] = -dE/dz_j = delta * old_w2[j] * h[j] * (1 - h[j])
        for (int j = 0; j < num_hidden_; ++j)
        {
            double delta_h = delta * old_w2[j] * h[j] * (1.0 - h[j]);
            const int offset = j * num_in_;

            for (int i = 0; i < num_in_; ++i)
            {
                double grad_w1 = delta_h * x[i] - weight_decay_ * w1_[offset + i];
                v_w1_[offset + i] = momentum_ * v_w1_[offset + i] + alpha_ * grad_w1;
                w1_[offset + i] += v_w1_[offset + i];
            }

            v_b1_[j] = momentum_ * v_b1_[j] + alpha_ * delta_h;
            b1_[j] += v_b1_[j];
        }
    }

    void FeedForwardNet::write(std::ostream &out) const
    {
        out << num_in_ << ' ' << num_hidden_ << ' ' << alpha_ << ' '
            << momentum_ << ' ' << weight_decay_ << ' ' << gradient_clip_ << '\n';

        out << std::setprecision(17);
        for (double w : w1_)
            out << w << ' ';
        out << '\n';
        for (double b : b1_)
            out << b << ' ';
        out << '\n';
        for (double w : w2_)
            out << w << ' ';
        out << '\n';
        out << b2_ << '\n';
    }

    void FeedForwardNet::read(std::istream &in)
    {
        in >> num_in_ >> num_hidden_ >> alpha_ >> momentum_ >> weight_decay_ >> gradient_clip_;
        w1_.resize(num_hidden_ * num_in_);
        b1_.resize(num_hidden_);
        w2_.resize(num_hidden_);

        for (double &w : w1_)
            in >> w;
        for (double &b : b1_)
            in >> b;
        for (double &w : w2_)
            in >> w;
        in >> b2_;

        v_w1_.assign(w1_.size(), 0.0);
        v_b1_.assign(b1_.size(), 0.0);
        v_w2_.assign(w2_.size(), 0.0);
        v_b2_ = 0.0;
    }

    // ---------------------------------------------------------------------------
    // NeuralNet Approximator (Coordinator of 3-action value functions)
    // ---------------------------------------------------------------------------

    NeuralNet::NeuralNet(const Config_NN &config)
        : config_(config)
    {
        if (config_.architecture == NNArchitecture::SeparatePerAction)
        {
            subnets_.reserve(config_.num_of_actions);
            for (int a = 0; a < config_.num_of_actions; ++a)
            {
                // Unique seed per action sub-network
                uint64_t action_seed = config_.seed * 10007 + a * 31337 + 101;
                subnets_.emplace_back(config_.num_of_state_variables, config_.num_of_hidden,
                                      config_.alpha, config_.momentum, config_.weight_decay,
                                      config_.gradient_clip, action_seed);
            }
        }
        else
        {
            init_shared_weights(config_.seed);
        }
    }

    void NeuralNet::init_shared_weights(uint64_t seed)
    {
        int num_in = config_.num_of_state_variables;
        int num_hidden = config_.num_of_hidden;
        int num_act = config_.num_of_actions;

        shared_w1_.resize(num_hidden * num_in);
        shared_b1_.resize(num_hidden, 0.0);
        shared_w2_.resize(num_act * num_hidden);
        shared_b2_.resize(num_act, 0.0);

        v_shared_w1_.assign(shared_w1_.size(), 0.0);
        v_shared_b1_.assign(shared_b1_.size(), 0.0);
        v_shared_w2_.assign(shared_w2_.size(), 0.0);
        v_shared_b2_.assign(shared_b2_.size(), 0.0);

        std::mt19937_64 rng(seed);
        double lim1 = std::sqrt(6.0 / (num_in + num_hidden));
        std::uniform_real_distribution<double> d1(-lim1, lim1);
        for (double &w : shared_w1_)
            w = d1(rng);

        double lim2 = std::sqrt(6.0 / (num_hidden + 1));
        std::uniform_real_distribution<double> d2(-lim2, lim2);
        for (double &w : shared_w2_)
            w = d2(rng);
    }

    double NeuralNet::value(const std::vector<double> &x, int action) const
    {
        if (action < 0 || action >= config_.num_of_actions)
        {
            throw std::runtime_error("NeuralNet::value invalid action: " + std::to_string(action));
        }

        if (config_.architecture == NNArchitecture::SeparatePerAction)
        {
            return subnets_[action].predict(x);
        }
        return predict_shared(x, action);
    }

    void NeuralNet::update(const std::vector<double> &x, int action, double target)
    {
        if (action < 0 || action >= config_.num_of_actions)
        {
            throw std::runtime_error("NeuralNet::update invalid action: " + std::to_string(action));
        }

        if (config_.architecture == NNArchitecture::SeparatePerAction)
        {
            subnets_[action].train_step(x, target);
        }
        else
        {
            train_shared(x, action, target);
        }
    }

    double NeuralNet::predict_shared(const std::vector<double> &x, int action) const
    {
        int num_in = config_.num_of_state_variables;
        int num_hidden = config_.num_of_hidden;

        double out = shared_b2_[action];
        const int w2_offset = action * num_hidden;

        for (int j = 0; j < num_hidden; ++j)
        {
            double z = shared_b1_[j];
            const int w1_offset = j * num_in;
            for (int i = 0; i < num_in; ++i)
            {
                z += shared_w1_[w1_offset + i] * x[i];
            }
            double h = sigmoid(z);
            out += shared_w2_[w2_offset + j] * h;
        }

        return out;
    }

    void NeuralNet::train_shared(const std::vector<double> &x, int action, double target)
    {
        int num_in = config_.num_of_state_variables;
        int num_hidden = config_.num_of_hidden;

        // 1. Forward pass
        std::vector<double> h(num_hidden);
        double y_hat = shared_b2_[action];
        const int w2_offset = action * num_hidden;

        for (int j = 0; j < num_hidden; ++j)
        {
            double z = shared_b1_[j];
            const int w1_offset = j * num_in;
            for (int i = 0; i < num_in; ++i)
            {
                z += shared_w1_[w1_offset + i] * x[i];
            }
            h[j] = sigmoid(z);
            y_hat += shared_w2_[w2_offset + j] * h[j];
        }

        // 2. TD Error delta
        double delta = target - y_hat;
        if (config_.gradient_clip > 0.0)
        {
            delta = std::clamp(delta, -config_.gradient_clip, config_.gradient_clip);
        }

        // 3. Update action output weights
        std::vector<double> old_w2_action(num_hidden);
        for (int j = 0; j < num_hidden; ++j)
        {
            old_w2_action[j] = shared_w2_[w2_offset + j];
            double grad_w2 = delta * h[j] - config_.weight_decay * shared_w2_[w2_offset + j];
            v_shared_w2_[w2_offset + j] = config_.momentum * v_shared_w2_[w2_offset + j] + config_.alpha * grad_w2;
            shared_w2_[w2_offset + j] += v_shared_w2_[w2_offset + j];
        }
        v_shared_b2_[action] = config_.momentum * v_shared_b2_[action] + config_.alpha * delta;
        shared_b2_[action] += v_shared_b2_[action];

        // 4. Backprop to shared hidden weights
        for (int j = 0; j < num_hidden; ++j)
        {
            double delta_h = delta * old_w2_action[j] * h[j] * (1.0 - h[j]);
            const int w1_offset = j * num_in;

            for (int i = 0; i < num_in; ++i)
            {
                double grad_w1 = delta_h * x[i] - config_.weight_decay * shared_w1_[w1_offset + i];
                v_shared_w1_[w1_offset + i] = config_.momentum * v_shared_w1_[w1_offset + i] + config_.alpha * grad_w1;
                shared_w1_[w1_offset + i] += v_shared_w1_[w1_offset + i];
            }

            v_shared_b1_[j] = config_.momentum * v_shared_b1_[j] + config_.alpha * delta_h;
            shared_b1_[j] += v_shared_b1_[j];
        }
    }

    void NeuralNet::set_learning_rate(double alpha)
    {
        config_.alpha = alpha;
        for (auto &subnet : subnets_)
        {
            subnet.set_alpha(alpha);
        }
    }

    void NeuralNet::save(const std::string &path) const
    {
        std::ofstream out(path);
        if (!out)
        {
            throw std::runtime_error("Cannot open file for writing NeuralNet: " + path);
        }

        out << "# NeuralNet Model\n";
        out << config_.num_of_state_variables << ' '
            << config_.num_of_hidden << ' '
            << config_.num_of_actions << ' '
            << config_.alpha << ' '
            << config_.momentum << ' '
            << config_.weight_decay << ' '
            << config_.gradient_clip << ' '
            << static_cast<int>(config_.architecture) << '\n';

        if (config_.architecture == NNArchitecture::SeparatePerAction)
        {
            for (const auto &subnet : subnets_)
            {
                subnet.write(out);
            }
        }
        else
        {
            out << std::setprecision(17);
            for (double w : shared_w1_)
                out << w << ' ';
            out << '\n';
            for (double b : shared_b1_)
                out << b << ' ';
            out << '\n';
            for (double w : shared_w2_)
                out << w << ' ';
            out << '\n';
            for (double b : shared_b2_)
                out << b << ' ';
            out << '\n';
        }
    }

    void NeuralNet::load(const std::string &path)
    {
        std::ifstream in(path);
        if (!in)
        {
            throw std::runtime_error("Cannot open file for loading NeuralNet: " + path);
        }

        std::string line;
        while (std::getline(in, line))
        {
            if (!line.empty() && line[0] != '#')
            {
                break;
            }
        }

        std::istringstream iss(line);
        int sv, hidden, actions, arch_int;
        double alpha, momentum, wd, clip;
        if (!(iss >> sv >> hidden >> actions >> alpha >> momentum >> wd >> clip >> arch_int))
        {
            throw std::runtime_error("Invalid NeuralNet checkpoint header: " + path);
        }

        if (sv != config_.num_of_state_variables || hidden != config_.num_of_hidden ||
            actions != config_.num_of_actions)
        {
            throw std::runtime_error("NeuralNet architecture parameter mismatch in: " + path);
        }

        config_.alpha = alpha;
        config_.momentum = momentum;
        config_.weight_decay = wd;
        config_.gradient_clip = clip;
        config_.architecture = static_cast<NNArchitecture>(arch_int);

        if (config_.architecture == NNArchitecture::SeparatePerAction)
        {
            subnets_.clear();
            for (int a = 0; a < config_.num_of_actions; ++a)
            {
                FeedForwardNet net;
                net.read(in);
                subnets_.push_back(net);
            }
        }
        else
        {
            shared_w1_.resize(config_.num_of_hidden * config_.num_of_state_variables);
            shared_b1_.resize(config_.num_of_hidden);
            shared_w2_.resize(config_.num_of_actions * config_.num_of_hidden);
            shared_b2_.resize(config_.num_of_actions);

            for (double &w : shared_w1_)
                in >> w;
            for (double &b : shared_b1_)
                in >> b;
            for (double &w : shared_w2_)
                in >> w;
            for (double &b : shared_b2_)
                in >> b;

            v_shared_w1_.assign(shared_w1_.size(), 0.0);
            v_shared_b1_.assign(shared_b1_.size(), 0.0);
            v_shared_w2_.assign(shared_w2_.size(), 0.0);
            v_shared_b2_.assign(shared_b2_.size(), 0.0);
        }
    }

} // namespace keepaway
