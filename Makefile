# Keepaway term project
#   make                                   build every program into build/
#   ./build/run_benchmarks 1000            benchmark policies -> results/benchmarks.csv
#   ./build/run_benchmarks 200 1 --log results/logs
#   python3 tools/viewer.py results/logs/traj_HandCoded.csv
#
# Layout
#   src/sim/     simulator (keepaway_env)
#   src/learn/   learning code: approximator interface, Q-learning, tile coding, RBF, neural net
#   apps/        one .cpp with main() per program; each one builds to build/<name>
#   tools/       Python scripts (viewer, plots)
#   results/     CSVs and logs written by the programs (not committed)

CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
CPPFLAGS += -Isrc
BUILD := build

LIB_SRC := $(wildcard src/sim/*.cpp src/learn/*.cpp)
LIB_OBJ := $(patsubst src/%.cpp,$(BUILD)/obj/%.o,$(LIB_SRC))
HEADERS := $(wildcard src/sim/*.hpp src/learn/*.hpp)
APPS := $(patsubst apps/%.cpp,$(BUILD)/%,$(wildcard apps/*.cpp))

all: $(APPS)

$(BUILD)/obj/%.o: src/%.cpp $(HEADERS)
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD)/%: apps/%.cpp $(LIB_OBJ) $(HEADERS)
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LIB_OBJ) -o $@

clean:
	rm -rf $(BUILD)

# Keep object files between builds (make would otherwise delete them as intermediates).
.SECONDARY: $(LIB_OBJ)

.PHONY: all clean
