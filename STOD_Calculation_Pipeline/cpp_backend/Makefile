CXX = g++
# Note: Canary doesn't need HDF5, just the logic header.
# Optimization flags:
#   -O3: Maximum optimization level
#   -march=native: Optimize for the specific CPU architecture
#   -ffast-math: Aggressive floating-point optimizations (safe for physics simulations)
#   -funroll-loops: Unroll small loops for better performance
#   -flto: Link-time optimization for cross-file inlining
CXXFLAGS = -O3 -fopenmp -std=c++17 -march=native -ffast-math -funroll-loops -flto

# Paths for HDF5 (needed for main tools)
# Check if EBROOTHDF5 is set (from module load), otherwise try to find HDF5
ifeq ($(EBROOTHDF5),)
    # Try to use pkg-config if available
    ifneq ($(shell which pkg-config 2>/dev/null),)
        HDF5_INC = $(shell pkg-config --cflags hdf5 2>/dev/null || echo "")
        HDF5_LIB = $(shell pkg-config --libs hdf5 2>/dev/null || echo "-lhdf5_cpp -lhdf5")
    else
        # Fallback: try common locations
        HDF5_INC = -I/include
        HDF5_LIB = -L/lib -lhdf5_cpp -lhdf5
    endif
    # Check if HDF5 headers can be found
    HDF5_CHECK = $(shell test -f $(EBROOTHDF5)/include/H5Cpp.h 2>/dev/null && echo "found" || echo "notfound")
else
    HDF5_INC = -I$(EBROOTHDF5)/include 
    HDF5_LIB = -L$(EBROOTHDF5)/lib -lhdf5_cpp -lhdf5
endif

ALL_TARGETS = gen_traj calc_metrics sod_canary

all: check_hdf5 $(ALL_TARGETS)

check_hdf5:
	@if [ -z "$(EBROOTHDF5)" ]; then \
		echo "WARNING: EBROOTHDF5 not set. Make sure HDF5 module is loaded:"; \
		echo "  module load HDF5/1.14.0-gompi-2022b"; \
	fi

gen_traj: gen_traj.cpp systems.hpp integrator.hpp
	$(CXX) $(CXXFLAGS) $(HDF5_INC) gen_traj.cpp -o gen_traj $(HDF5_LIB)

calc_metrics: calc_metrics.cpp sod_logic.hpp systems.hpp integrator.hpp
	$(CXX) $(CXXFLAGS) $(HDF5_INC) calc_metrics.cpp -o calc_metrics $(HDF5_LIB)

sod_canary: sod_canary.cpp sod_logic.hpp
	$(CXX) $(CXXFLAGS) sod_canary.cpp -o sod_canary

clean:
	rm -f $(ALL_TARGETS)

