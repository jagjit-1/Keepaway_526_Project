// approximator.hpp
// Interface shared by every value-function approximator: tile coding (CMAC), RBF and the
// neural network. The learning loop only talks to this interface, so approximators can be
// swapped without touching the learning code.

#pragma once
#include <string>
#include <vector>

namespace keepaway {

class Approximator {
 public:
  virtual ~Approximator() = default;
  // Estimated value of taking `action` in the state described by features x.
  virtual double value(const std::vector<double>& x, int action) const = 0;
  // Moves value(x, action) toward `target` (learning rate is internal to each approximator).
  virtual void update(const std::vector<double>& x, int action, double target) = 0;
  virtual void save(const std::string& path) const = 0;
  virtual void load(const std::string& path) = 0;
};

}  // namespace keepaway
