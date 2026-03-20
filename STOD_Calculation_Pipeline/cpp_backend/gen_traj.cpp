#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <omp.h>
#include <atomic>
#include <chrono>
#include <iomanip>
#include "H5Cpp.h"
#include "systems.hpp"
#include "integrator.hpp"

using namespace H5;
using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[]) {
    // Suppress HDF5 error printing
    H5::Exception::dontPrint();

    try {
        // We added 3 args for flags: do_fwd, do_bwd, use_orthogonal
        if (argc < 16) { 
            cerr << "Usage: gen_traj <sys> <nx> <ny> <x_min> <x_max> <y_min> <y_max> <t_end> <dt> <r_start> <r_end> <outfile> <do_fwd> <do_bwd> <use_orthogonal> [params...]" << endl;
            return 1;
        }

        int idx = 1;
        string sys_name = argv[idx++];
        int nx = stoi(argv[idx++]);
        int ny = stoi(argv[idx++]);
        double x_min = stod(argv[idx++]);
        double x_max = stod(argv[idx++]);
        double y_min = stod(argv[idx++]);
        double y_max = stod(argv[idx++]);
        double t_end = stod(argv[idx++]); 
        double dt    = stod(argv[idx++]);
        int r_start  = stoi(argv[idx++]);
        int r_end    = stoi(argv[idx++]);
        string out_filename = argv[idx++];
        
        // --- Read Direction Toggles ---
        int do_fwd = stoi(argv[idx++]);
        int do_bwd = stoi(argv[idx++]);
        
        // --- Read Orthogonal Mode Toggle ---
        int use_orthogonal = stoi(argv[idx++]);

        // Parse Physics Parameters
        map<string, double> phys_params;
        for (int i = idx; i < argc; ++i) {
            string arg = argv[i];
            size_t eq_pos = arg.find('=');
            if (eq_pos != string::npos) {
                try {
                    phys_params[arg.substr(0, eq_pos)] = stod(arg.substr(eq_pos + 1));
                } catch (...) {}
            }
        }
        
        // --- 1. Instantiate System (to get dimensions) ---
        DynamicalSystem* test_sys = create_system(sys_name, phys_params);
        if(!test_sys) { 
            cerr << "Error: Unknown system module '" << sys_name << "'" << endl; 
            return 1; 
        }
        int n_dims = test_sys->get_dim();
        delete test_sys;

        // --- 2. HDF5 Setup ---
        int rows_to_proc = r_end - r_start;
        int steps = (int)(ceil(t_end / dt)) + 1;

        H5File file(out_filename, H5F_ACC_TRUNC);
        
        hsize_t dimsf[4] = {
            (hsize_t)rows_to_proc, 
            (hsize_t)nx, 
            (hsize_t)steps, 
            (hsize_t)n_dims
        };
        
        DSetCreatPropList plist;
        hsize_t chunk_dims[4] = {1, (hsize_t)nx, (hsize_t)steps, (hsize_t)n_dims};
        plist.setChunk(4, chunk_dims);
        plist.setDeflate(4); 

        // We always create the datasets (filled with 0) so the reader doesn't crash,
        // but we only populate them if enabled.
        DataSpace dataspace(4, dimsf);
        DataSet dataset_fwd = file.createDataSet("forward", PredType::NATIVE_DOUBLE, dataspace, plist);
        DataSet dataset_bwd = file.createDataSet("backward", PredType::NATIVE_DOUBLE, dataspace, plist);

        // --- 3. Parallel Computation ---
        size_t total_elems = (size_t)rows_to_proc * nx * steps * n_dims;
        
        // Initialize with zeros
        vector<double> buffer_fwd(total_elems, 0.0);
        vector<double> buffer_bwd(total_elems, 0.0);

        // Progress tracking
        size_t total_cells = (size_t)rows_to_proc * nx;
        atomic<size_t> cells_completed(0);
        atomic<size_t> cells_processed(0);  // Separate counter for actual integrations
        atomic<int> last_reported_percent(-1);
        auto start_time = high_resolution_clock::now();
        atomic<long long> last_report_time_seconds(0);  // Store as seconds since start
        const int report_interval_seconds = 5; // Report every 5 seconds minimum

        cout << "=======================================================================" << endl;
        cout << "Starting Trajectory Generation" << endl;
        cout << "  System: " << sys_name << endl;
        cout << "  Rows: " << r_start << " to " << (r_end - 1) << " (processing " << rows_to_proc << " rows)" << endl;
        cout << "  Columns: " << nx << endl;
        cout << "  Total cells: " << total_cells << endl;
        cout << "  Time range: 0.0 to " << t_end << " (dt=" << dt << ", " << steps << " steps)" << endl;
        cout << "  Forward: " << (do_fwd ? "YES" : "NO") << ", Backward: " << (do_bwd ? "YES" : "NO") << endl;
        cout << "  Field mode: " << (use_orthogonal ? "ORTHOGONAL" : "STANDARD") << endl;
        cout << "  Threads: " << omp_get_max_threads() << endl;
        cout << "=======================================================================" << endl;
        cout << "[Progress] Starting... (will report every 5 seconds or 1% progress)" << endl;
        cout.flush();

        #pragma omp parallel
        {
            DynamicalSystem* sys = create_system(sys_name, phys_params);
            
            vector<double> traj_cache; 
            vector<double> init_state;

            #pragma omp for schedule(dynamic)
            for (int r = 0; r < rows_to_proc; ++r) {
                int abs_row = r_start + r;
                double grid_v = y_min + (abs_row + 0.5) * (y_max - y_min) / ny;

                for (int c = 0; c < nx; ++c) {
                    double grid_u = x_min + (c + 0.5) * (x_max - x_min) / nx;
                    
                    size_t offset = ((size_t)r * nx * steps * n_dims) + ((size_t)c * steps * n_dims);

                    // Only integrate if the system says the initial state is valid
                    // (e.g. LiCN energy check happens here)
                    if (sys->get_initial_state(grid_u, grid_v, init_state)) {
                        cells_processed.fetch_add(1, memory_order_relaxed);
                        
                        // --- FORWARD LOGIC ---
                        if (do_fwd) {
                            traj_cache.clear();
                            // Integrate to +t_end (use orthogonal field if flag is set)
                            integrate_path_adaptive(sys, init_state, 0.0, t_end, dt, 1.0, traj_cache, use_orthogonal != 0);
                            
                            size_t fwd_limit = min(traj_cache.size(), (size_t)(steps * n_dims));
                            for(size_t k=0; k < fwd_limit; ++k) {
                                buffer_fwd[offset + k] = traj_cache[k];
                            }
                        }

                        // --- BACKWARD LOGIC ---
                        if (do_bwd) {
                            traj_cache.clear();
                            // Integrate to -t_end (Direction -1.0, use orthogonal field if flag is set)
                            integrate_path_adaptive(sys, init_state, t_end, 0.0, dt, -1.0, traj_cache, use_orthogonal != 0);
                            
                            size_t bwd_limit = min(traj_cache.size(), (size_t)(steps * n_dims));
                            for(size_t k=0; k < bwd_limit; ++k) {
                                buffer_bwd[offset + k] = traj_cache[k];
                            }
                        }
                    } 
                }

                // Progress reporting (thread-safe)
                cells_completed.fetch_add(nx, memory_order_relaxed);
                
                // Check if we should report (simplified logic)
                size_t completed = cells_completed.load(memory_order_relaxed);
                int current_percent = (int)(100.0 * completed / total_cells);
                auto now = high_resolution_clock::now();
                auto total_elapsed_sec = duration_cast<seconds>(now - start_time).count();
                long long last_report_sec = last_report_time_seconds.load(memory_order_relaxed);
                long long elapsed_since_report = total_elapsed_sec - last_report_sec;
                
                // Report if: 1) 5+ seconds passed, OR 2) 1% more progress made, OR 3) first/last row
                bool should_report = false;
                if (elapsed_since_report >= report_interval_seconds) {
                    // Time-based: try to update the report time atomically
                    long long expected = last_report_sec;
                    if (last_report_time_seconds.compare_exchange_weak(expected, total_elapsed_sec, memory_order_relaxed)) {
                        should_report = true;
                    }
                } else if (current_percent > last_reported_percent.load(memory_order_relaxed)) {
                    // Percent-based: report every 1% progress
                    int expected = last_reported_percent.load();
                    if (last_reported_percent.compare_exchange_weak(expected, current_percent, memory_order_relaxed)) {
                        should_report = true;
                    }
                } else if (r == 0 || r == rows_to_proc - 1) {
                    // Always report first and last row
                    should_report = true;
                }
                
                if (should_report) {
                    size_t processed = cells_processed.load(memory_order_relaxed);
                    double percent = 100.0 * completed / total_cells;
                    
                    double rate = (total_elapsed_sec > 0) ? (double)completed / total_elapsed_sec : 0.0;
                    size_t remaining = total_cells - completed;
                    int eta_seconds = (rate > 0) ? (int)(remaining / rate) : 0;
                    
                    int total_h = total_elapsed_sec / 3600;
                    int total_m = (total_elapsed_sec % 3600) / 60;
                    int total_s = total_elapsed_sec % 60;
                    
                    int eta_h = eta_seconds / 3600;
                    int eta_m = (eta_seconds % 3600) / 60;
                    int eta_s = eta_seconds % 60;
                    
                    cout << "[Progress] Row " << abs_row << "/" << (r_end - 1) 
                         << " | Cells: " << completed << "/" << total_cells 
                         << " (" << fixed << setprecision(1) << percent << "%)"
                         << " | Processed: " << processed
                         << " | Rate: " << setprecision(1) << rate << " cells/s"
                         << " | Elapsed: " << setfill('0') << setw(2) << total_h << ":" 
                         << setfill('0') << setw(2) << total_m << ":" 
                         << setfill('0') << setw(2) << total_s
                         << " | ETA: " << setfill('0') << setw(2) << eta_h << ":" 
                         << setfill('0') << setw(2) << eta_m << ":" 
                         << setfill('0') << setw(2) << eta_s << endl;
                    cout.flush();
                }
            }
            delete sys;
        }

        auto end_time = high_resolution_clock::now();
        auto total_duration = duration_cast<seconds>(end_time - start_time).count();
        int total_h = total_duration / 3600;
        int total_m = (total_duration % 3600) / 60;
        int total_s = total_duration % 60;
        
        cout << "=======================================================================" << endl;
        cout << "Trajectory Generation Complete!" << endl;
        cout << "  Total time: " << setfill('0') << setw(2) << total_h << ":" 
             << setfill('0') << setw(2) << total_m << ":" 
             << setfill('0') << setw(2) << total_s << endl;
        cout << "  Output file: " << out_filename << endl;
        cout << "=======================================================================" << endl;
        cout.flush();

        // --- 4. Write to Disk ---
        // Always write the buffers. If a direction was disabled, it writes zeros.
        dataset_fwd.write(buffer_fwd.data(), PredType::NATIVE_DOUBLE);
        dataset_bwd.write(buffer_bwd.data(), PredType::NATIVE_DOUBLE);
        
        file.close();

    } catch(const Exception& e) {
        cerr << "HDF5 Error: ";
        e.printErrorStack();
        return 1;
    } catch(const exception& e) {
        cerr << "Standard Exception: " << e.what() << endl;
        return 1;
    } catch(...) {
        cerr << "Unknown Error in gen_traj" << endl;
        return 1;
    }
    return 0;
}


