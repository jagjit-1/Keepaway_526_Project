// ---------------------------------------------------------------------------
// neural_net.hpp
// Neural network function approximator for Keepaway reinforcement learning.
// Implements the Approximator interface.
//
// Based on:
//   Stone, Kuhlmann, Taylor & Liu (2005),
//   "Keepaway Soccer: From Machine Learning Testbed to Benchmark", Section 4.3:
//   "We use three separate 13-20-1 networks, one for each action.
//    Inputs to the neural network are set to the value of state variables
//    and each network's output corresponds to an action. Nodes in the hidden
//    layer have a sigmoid transfer function and output nodes are linear.
//    We use standard backpropagation to modify the weights in the neural
//    networks to back up Sarsa(lambda) / Q-learning values."
// ---------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "approximator.hpp"

namespace keepaway
{

    enum class NNArchitecture
    {
        // Stone et al. (2005): 3 separate 13-20-1 networks, one per action
        SeparatePerAction = 0,
        // Shared hidden layer with 3 linear outputs: 13-20-3
        SharedHidden = 1
    };

    struct Config_NN
    {
        int num_of_state_variables = 13; // default: 13 continuous state features
        int num_of_hidden = 20;          // 20 hidden units (paper: 13-20-1)
        int num_of_actions = 3;          // 3 actions: HOLD, PASS_F2, PASS_F3
        double alpha = 0.01;             // learning rate for gradient descent
        double momentum = 0.0;           // momentum factor [0, 1)
        double weight_decay = 0.0;       // L2 weight regularization
        double gradient_clip = 1.0;      // max absolute TD error for backprop clipping (0 = off)
        uint64_t seed = 42;              // seed for reproducible weight initialization
        NNArchitecture architecture = NNArchitecture::SeparatePerAction;
    };

    // Single feedforward sub-network: N_in -> N_hidden (sigmoid) -> 1 (linear).
    class FeedForwardNet
    {
    public:
        FeedForwardNet() = default;
        FeedForwardNet(int num_in, int num_hidden, double alpha, double momentum,
                       double weight_decay, double gradient_clip, uint64_t seed);

        // Forward pass: returns linear output for input vector x.
        double predict(const std::vector<double> &x) const;

        // Forward pass caching activations, followed by backprop for TD-target y.
        void train_step(const std::vector<double> &x, double target);

        // Serialization
        void write(std::ostream &out) const;
        void read(std::istream &in);

        int num_inputs() const { return num_in_; }
        int num_hidden() const { return num_hidden_; }
        double alpha() const { return alpha_; }
        void set_alpha(double a) { alpha_ = a; }

    private:
        int num_in_ = 0;
        int num_hidden_ = 0;
        double alpha_ = 0.01;
        double momentum_ = 0.0;
        double weight_decay_ = 0.0;
        double gradient_clip_ = 1.0;

        // Weights & biases
        // W1: [num_hidden_ * num_in_], b1: [num_hidden_]
        std::vector<double> w1_;
        std::vector<double> b1_;
        // W2: [num_hidden_], b2: scalar
        std::vector<double> w2_;
        double b2_ = 0.0;

        // Momentum velocities
        std::vector<double> v_w1_;
        std::vector<double> v_b1_;
        std::vector<double> v_w2_;
        double v_b2_ = 0.0;

        void init_weights(uint64_t seed);
    };

    // Full Approximator implementing the 3-action Keepaway value function.
    class NeuralNet : public Approximator
    {
    public:
        explicit NeuralNet(const Config_NN &config);

        // Estimated value Q(x, action)
        double value(const std::vector<double> &x, int action) const override;

        // Moves Q(x, action) toward target via backpropagation
        void update(const std::vector<double> &x, int action, double target) override;

        // Model checkpointing
        void save(const std::string &path) const override;
        void load(const std::string &path) override;

        const Config_NN &config() const { return config_; }
        void set_learning_rate(double alpha);

    private:
        Config_NN config_;
        // One separate sub-network per action (Stone et al. 2005)
        std::vector<FeedForwardNet> subnets_;

        // For SharedHidden architecture:
        // W1 shared [num_hidden * num_in], b1 [num_hidden]
        // W2 per action [num_actions * num_hidden], b2 per action [num_actions]
        std::vector<double> shared_w1_;
        std::vector<double> shared_b1_;
        std::vector<double> shared_w2_;
        std::vector<double> shared_b2_;
        std::vector<double> v_shared_w1_;
        std::vector<double> v_shared_b1_;
        std::vector<double> v_shared_w2_;
        std::vector<double> v_shared_b2_;

        void init_shared_weights(uint64_t seed);
        double predict_shared(const std::vector<double> &x, int action) const;
        void train_shared(const std::vector<double> &x, int action, double target);
    };

} // namespace keepaway
