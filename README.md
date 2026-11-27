# 3-v-2 Keepaway Reinforcement Learning

A comprehensive C++ reinforcement learning simulator and function approximator suite for the **3-v-2 Keepaway** soccer benchmark.

This project implements, evaluates, and compares multiple value-function approximation methods under Semi-Markov Decision Process (SMDP) Q-learning:
1. **Neural Network (TD Backpropagation MLP)**: 3 separate 13–20–1 action networks or 13–20–3 shared-hidden-layer network.
2. **Tile Coding (CMAC)**: Linear function approximation over multi-dimensional tilings.
3. **Radial Basis Function (RBF)**: Gaussian basis function representation with normalized activations.
4. **Hand-Coded / Heuristic Baselines**: `AlwaysHold`, `Random`, and expert `HandCoded` policies.

---

## Declaration of LLM / AI Usage

In compliance with academic integrity guidelines:
- **Partial AI Generation**: Code structure, initial implementations of simulators/neural network routines, boilerplate build scripts, and initial draft visualizations were generated or refactored with the assistance of Large Language Models (Anthropic Claude & Google DeepMind Antigravity).
- **Human Author Contribution**: Problem formulation, parameter tuning, experimental runs, verification, debugging of compiler/environment configurations, integration of true $\epsilon$-greedy exploration into the Q-learning loop, and analysis of benchmark results were conducted by the student/authors.
- **Per-file notes**: each source file that an LLM wrote or changed says so in its header, with what was changed. For example, the merged training loop in `src/learn/train.cpp`, the command-line version of `apps/run_q_learning.cpp`, `tools/run_experiments.sh` and the `run_q_learning` part of "Building and Running" below were written by Claude (Anthropic) on 2026-11-26 at the request of Jagjit.

---

## Reinforcement Learning Method

### 1. SMDP Q-Learning Framework
In 3-v-2 Keepaway, keepers aim to keep possession of the ball in a designated area while two takers attempt to steal it.
The learner acts as a **Q-value function approximator** for the current ball holder:
- **Off-Policy Control**: The TD target is computed using $\max_{a'} Q(s', a')$.
- **Macro-Actions / Semi-Markov Decision Process (SMDP)**: Actions (`HOLD`, `PASS_F2`, `PASS_F3`) take variable continuous simulator time steps ($k \ge 1$) to execute.
- **Sparse Rewards & Discounting**:
  - Reward is 0 while play continues, and -1 when the episode terminates (turnover or out-of-bounds).
  - Terminal target: $G = -\gamma^{k-1}$
  - Non-terminal target: $G = \gamma^k \max_{a'} Q(s', a')$
- **Parameter Update**:
  $$w \leftarrow w + \alpha \left(G - Q(s, a)\right) \nabla_w Q(s, a)$$

### 2. Action Selection: $\epsilon$-Greedy Exploration vs. Greedy Caveat
- **The Greedy Caveat in Tabular/Linear RL**: When initial weights are zeros and actions are chosen purely greedily, the agent is susceptible to starvation of non-greedy actions and may fail to discover effective passing opportunities.
- **Solution Implemented**: The training engine now supports true **$\epsilon$-greedy exploration** (`epsilon_greedy()`), picking random exploratory actions with probability $\epsilon$ (default $\epsilon = 0.05$) during training, while keeping greedy action selection for evaluation.

---

## Function Approximators

### 1. Neural Network (TD Backpropagation MLP)
- **State Input**: 13 continuous keeper-relative features normalized to $[0, 1]$.
- **Architecture**:
  - **Default (Separate)**: 3 independent 13–20–1 networks, one per action. Eliminates cross-action weight interference.
  - **Shared Hidden Layer**: One 13–20–3 network with shared representations across all 3 actions.
- **Hidden Units**: 20 units with numerically stable sigmoid activation clamped to $[-45, 45]$.
- **Weights Initialization**: Xavier / Glorot uniform initialization.

### 2. Tile Coding (CMAC)
- Discretizes the 13-dimensional continuous state across multiple offset tilings.
- For each state variable, tiling, tile, and action, an independent weight is stored.
- Value estimate:
  $$Q(s, a) = \sum_{j \in \text{active}(s, a)} w_j$$
- Typical settings: 13 state variables, 10 tilings, 10 to 20 tiles, $\alpha = \frac{0.6}{13 \times 10}$, $\gamma = 0.99$.

### 3. Radial Basis Function (RBF)
- Continuous Gaussian basis activations centered across the feature space:
  $$\phi_{f, t}(x) = \exp\left(-\frac{(x_f - \mu_t)^2}{2\sigma^2}\right)$$
- Normalized update rule:
  $$\alpha_{\text{feature}} = \frac{\alpha}{\sum_t \phi_t(s)^2}, \quad w_{f, t, a} \leftarrow w_{f, t, a} + \alpha_{\text{feature}} (G - Q(s, a)) \phi_t(s)$$
- Typical settings: 200 centers per feature, $\sigma = 0.01$, $\alpha = \frac{0.6}{13}$, $\gamma = 0.99$.

---

## Directory Structure

```text
├── apps/
│   ├── common.hpp              # Shared logging, stats, CSV export
│   ├── run_benchmarks.cpp      # Baseline evaluation (AlwaysHold, Random, HandCoded)
│   ├── run_q_learning.cpp      # Tile coding & RBF trainer (command-line options, Q-learning or Sarsa)
│   ├── replay_rbf.cpp          # Plays saved RBF weights greedily and logs positions for viewer.py
│   └── train_neural_net.cpp    # Standalone Neural Net trainer & evaluator
├── src/
│   ├── learn/
│   │   ├── approximator.hpp    # Polymorphic Approximator interface
│   │   ├── neural_net.hpp      # Neural network headers and architecture structs
│   │   ├── neural_net.cpp      # MLP forward/backward pass & model persistence
│   │   ├── q_learning.hpp      # Tile coding and RBF class declarations
│   │   ├── rbf.cpp             # Radial basis function implementation
│   │   ├── tile_coding.cpp     # Tile coding (CMAC) implementation
│   │   ├── train.hpp           # Training & greedy evaluation loop interfaces
│   │   └── train.cpp           # One training loop (Q-learning or Sarsa, epsilon-greedy) and greedy evaluation
│   └── sim/
│       ├── keepaway_env.hpp    # 3-v-2 continuous physics environment definitions
│       └── keepaway_env.cpp    # Environment kinematics, state normalization, policies
├── tools/
│   ├── plot_nn_results.py      # Plotting script for neural network benchmarks
│   ├── plot_results.py         # Plotting script for TC and RBF sweeps
│   ├── plot_final.py           # Final TC / RBF / NN comparison and learned-weights figures
│   ├── run_experiments.sh      # One command per tile coding / RBF experiment case
│   └── viewer.py               # Trajectory visualizer
├── results/
│   ├── figures/                # Output publication figures (PNG)
│   └── nn/                     # Checkpoints and CSV logs from neural net runs
├── Makefile                    # Multi-target build script
└── README.md                   # This project guide
```

---

## Building and Running

### 1. Build
```bash
make
```

### 2. Training the Neural Network
Run with Q-learning ($\epsilon$-greedy off-policy control):
```bash
./build/train_neural_net --algo qlearn --episodes 1000 --eval-episodes 200 --alpha 0.01 --hidden 20 --epsilon 0.05
```

Run with Sarsa:
```bash
./build/train_neural_net --algo sarsa --episodes 1000 --eval-episodes 200 --alpha 0.01 --hidden 20 --epsilon 0.02
```

By default all three keepers learn, as in the tile coding / RBF runs (`--learners 3`);
`--learners 1` trains one keeper with two hand-coded teammates.

Run using the 13–20–3 Shared-Hidden-Layer architecture:
```bash
./build/train_neural_net --algo qlearn --shared --episodes 1000 --eval-episodes 200
```

### 3. Running Baselines & Tile Coding / RBF
```bash
# Run heuristic baselines:
./build/run_benchmarks 200 1

# Tile coding or RBF with the final settings (Q-learning, 13 state variables, 10 seeds):
./build/run_q_learning --approx tc
./build/run_q_learning --approx rbf

# Same with Sarsa, or with the 4 reduced state variables:
./build/run_q_learning --approx tc --algo sarsa
./build/run_q_learning --approx rbf --features reduced

# All options and their defaults:
./build/run_q_learning --help

# The experiment cases (tuning sweeps and final comparisons), one command each:
tools/run_experiments.sh list
tools/run_experiments.sh tc_tile_sweep
```
Each run writes `results/<name>_<timestamp>.csv` and the weights of every seed to
`results/weights/<name>/seed-<seed>.txt`. `--alpha` is the fraction of the TD error moved per
update; the program divides it by the number of active features.

### 4. Generating Visualizations
```bash
python tools/plot_nn_results.py
```
Generated plots are placed in `results/figures/`:
- `nn_learning_curves.png`: Moving average possession time over training episodes.
- `nn_policy_comparison.png`: Mean duration vs. baselines.
- `nn_termination_reasons.png`: Distribution of steals vs out-of-bounds across policies.

---

## Experimental Benchmark Results

Evaluated over 200 unseen test episodes (greedy evaluation):

| Policy / Approximator | Architecture / Notes | Mean Possession Time | Median Time | Ball Stolen | Out of Bounds |
|---|---|---|---|---|---|
| **Always Hold** | Constant action baseline | 4.85 s $\pm$ 0.80 | 4.30 s | 200 | 0 |
| **Random Policy** | Uniform random selection | 10.90 s $\pm$ 2.97 | 10.90 s | 194 | 6 |
| **Neural Net (Shared)** | 13-20-3 Q-learning | 12.69 s $\pm$ 4.87 | 12.70 s | 197 | 3 |
| **Neural Net (Separate)** | 3 $\times$ 13-20-1 Q-learning | **13.77 s** $\pm$ 5.81 | 12.90 s | 197 | 3 |
| **Neural Net (Separate)** | 3 $\times$ 13-20-1 Sarsa | **13.83 s** $\pm$ 5.96 | 12.90 s | 197 | 3 |
| **Hand-Coded Heuristic** | Domain-expert rule set | 22.89 s $\pm$ 11.01 | 20.30 s | 175 | 25 |
