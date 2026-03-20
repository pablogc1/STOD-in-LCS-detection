#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <map>
#include <atomic>
#include <iomanip>
#include <chrono>
#include "H5Cpp.h"
#include "sod_logic.hpp"
#include "systems.hpp"
#include "integrator.hpp" 

using namespace H5;
using namespace std;

// --- UTILS ---
inline void get_state_at(const vector<double>& data, size_t r, size_t c, size_t s, 
                         size_t nx, size_t n_steps, size_t n_dims, vector<double>& out) {
    // Helper for STOD (reading stored data)
    // Indexing: row, col, step, dim
    size_t idx = r * nx * n_steps * n_dims + c * n_steps * n_dims + s * n_dims;
    for(size_t k=0; k<n_dims; ++k) out[k] = data[idx+k];
}

string format_duration(double seconds) {
    if (seconds < 0) seconds = 0;
    int h = (int)(seconds / 3600);
    int m = (int)((seconds - h * 3600) / 60);
    int s = (int)(seconds - h * 3600 - m * 60);
    stringstream ss;
    ss << setfill('0') << setw(2) << h << ":" 
       << setfill('0') << setw(2) << m << ":" 
       << setfill('0') << setw(2) << s;
    return ss.str();
}

// =========================================================
// 1. FRESH FTLE CALCULATION (From Pipeline 2)
// =========================================================
// Re-integrates locally for high precision gradients
void compute_fresh_ftle_slice(
    DynamicalSystem* sys,
    int rows, int cols,      // Local dimensions
    int start_row,           // Global offset for physics calc
    int nx_global, int ny_global,
    double x_min, double x_max,
    double y_min, double y_max,
    double t_start, double t_end,
    vector<double>& out_ftle,
    const string& label,
    bool use_orthogonal = false
) {
    double integration_time = t_end - t_start;
    if (abs(integration_time) < 1e-9) {
        fill(out_ftle.begin(), out_ftle.end(), 0.0);
        return;
    }

    double direction = (integration_time > 0) ? 1.0 : -1.0;
    double dx = (x_max - x_min) / nx_global;
    double dy = (y_max - y_min) / ny_global;

    cout << "[INFO] Starting " << label << " FRESH FTLE (" 
         << t_start << " -> " << t_end << ")..." << endl;

    atomic<int> completed_rows(0);
    int report_interval = max(1, rows / 10);
    auto t_timer_start = chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        vector<double> center_state(sys->get_dim());
        vector<double> neighbor_state(sys->get_dim());
        vector<double> traj_cache; 
        
        #pragma omp for schedule(dynamic)
        for (int r = 0; r < rows; ++r) {
            int r_global = start_row + r;
            double y0 = y_min + (r_global + 0.5) * dy;

            for (int c = 0; c < cols; ++c) {
                double x0 = x_min + (c + 0.5) * dx;
                double eps = 1e-4; 

                // Central difference stencil
                double xR = x0 + eps, yR = y0;
                double xL = x0 - eps, yL = y0;
                double xU = x0,       yU = y0 + eps;
                double xD = x0,       yD = y0 - eps;

                // Integration lambda
                auto integrate_point = [&](double ix, double iy, double& fx, double& fy) {
                    if(!sys->get_initial_state(ix, iy, center_state)) {
                        fx = fy = NAN; return;
                    }
                    traj_cache.clear();
                    integrate_path_adaptive(sys, center_state, t_start, t_end, abs(integration_time), direction, traj_cache, use_orthogonal);
                    if(traj_cache.empty()) { fx=fy=NAN; return; }
                    
                    size_t dim = sys->get_dim();
                    size_t last_idx = traj_cache.size() - dim;
                    for(size_t k=0; k<dim; ++k) neighbor_state[k] = traj_cache[last_idx + k];
                    sys->project_to_grid(neighbor_state, fx, fy);
                };

                double fxR, fyR, fxL, fyL, fxU, fyU, fxD, fyD;
                integrate_point(xR, yR, fxR, fyR);
                integrate_point(xL, yL, fxL, fyL);
                integrate_point(xU, yU, fxU, fyU);
                integrate_point(xD, yD, fxD, fyD);

                if (isnan(fxR) || isnan(fxL) || isnan(fxU) || isnan(fxD)) {
                    out_ftle[r * cols + c] = 0.0;
                    continue;
                }

                double dfx_dx = (fxR - fxL) / (2.0 * eps);
                double dfy_dx = (fyR - fyL) / (2.0 * eps);
                double dfx_dy = (fxU - fxD) / (2.0 * eps);
                double dfy_dy = (fyU - fyD) / (2.0 * eps);

                double c11 = dfx_dx*dfx_dx + dfy_dx*dfy_dx;
                double c12 = dfx_dx*dfx_dy + dfy_dx*dfy_dy; 
                double c22 = dfx_dy*dfx_dy + dfy_dy*dfy_dy;

                double tr = c11 + c22;
                double det = c11*c22 - c12*c12;
                double gap = sqrt(max(0.0, tr*tr - 4*det));
                double lam_max = (tr + gap) / 2.0;

                if (lam_max > 0)
                    out_ftle[r * cols + c] = log(lam_max) / (2.0 * abs(integration_time));
                else
                    out_ftle[r * cols + c] = 0.0;
            }

            // Progress reporting
            int cr = ++completed_rows;
            if (cr % report_interval == 0 || cr == rows) {
                #pragma omp critical
                {
                   auto t_now = chrono::high_resolution_clock::now();
                   double elapsed = chrono::duration<double>(t_now - t_timer_start).count();
                   double pct = (double)cr / rows * 100.0;
                   double eta = (elapsed / cr) * (rows - cr);
                   cout << "[PROGRESS] " << label << " FTLE: " << cr << "/" << rows 
                        << " (" << fixed << setprecision(1) << pct << "%) - ETA: " 
                        << format_duration(eta) << endl << flush;
                }
            }
        }
    }
}

// =========================================================
// 1b. VARIATIONAL FTLE CALCULATION (Recommended - no finite difference artifacts)
// =========================================================
// Uses the variational equations to compute the Cauchy-Green tensor exactly.
// This avoids the numerical artifacts that occur with finite differences near
// invariant manifolds and fixed points.
void compute_variational_ftle_slice(
    DynamicalSystem* sys,
    int rows, int cols,
    int start_row,
    int nx_global, int ny_global,
    double x_min, double x_max,
    double y_min, double y_max,
    double t_start, double t_end,
    double alpha_rescale,
    vector<double>& out_ftle,
    const string& label,
    bool use_orthogonal = false
) {
    double integration_time = t_end - t_start;
    if (abs(integration_time) < 1e-9) {
        fill(out_ftle.begin(), out_ftle.end(), 0.0);
        return;
    }

    double dx = (x_max - x_min) / nx_global;
    double dy = (y_max - y_min) / ny_global;

    cout << "[INFO] Starting " << label << " VARIATIONAL FTLE (" 
         << t_start << " -> " << t_end << ")..." << endl;

    atomic<int> completed_rows(0);
    int report_interval = max(1, rows / 10);
    auto t_timer_start = chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        vector<double> init_state(sys->get_dim());
        
        #pragma omp for schedule(dynamic)
        for (int r = 0; r < rows; ++r) {
            int r_global = start_row + r;
            double y0 = y_min + (r_global + 0.5) * dy;

            for (int c = 0; c < cols; ++c) {
                double x0 = x_min + (c + 0.5) * dx;
                
                if (!sys->get_initial_state(x0, y0, init_state)) {
                    out_ftle[r * cols + c] = 0.0;
                    continue;
                }

                // Use variational integrator - it computes the tangent matrix Phi
                // and derives FTLE from the Cauchy-Green tensor eigenvalue
                VariationalResult res = integrate_variational(
                    sys, init_state, t_start, t_end, alpha_rescale, 0.5, use_orthogonal
                );
                
                // FTLE = ftle_value / |T| (variational result gives log(sqrt(lambda_max)))
                out_ftle[r * cols + c] = res.ftle_value / abs(integration_time);
            }

            // Progress reporting
            int cr = ++completed_rows;
            if (cr % report_interval == 0 || cr == rows) {
                #pragma omp critical
                {
                   auto t_now = chrono::high_resolution_clock::now();
                   double elapsed = chrono::duration<double>(t_now - t_timer_start).count();
                   double pct = (double)cr / rows * 100.0;
                   double eta = (elapsed / cr) * (rows - cr);
                   cout << "[PROGRESS] " << label << " FTLE (var): " << cr << "/" << rows 
                        << " (" << fixed << setprecision(1) << pct << "%) - ETA: " 
                        << format_duration(eta) << endl << flush;
                }
            }
        }
    }
}

// =========================================================
// 2. FRESH ISTOD/STOD CALCULATION (New Methodology)
// =========================================================
// Fresh integration from t_snap (matching Python approach)
// STOD: forward paths (Level 0 = initial cell at t_snap)
// FINSTOD: reversed paths (Level 0 = final cell)
// Returns timing breakdown: trajectory_seconds, stod_seconds, finstod_seconds
struct SmodTimingBreakdown {
    double trajectory_seconds = 0.0;
    double stod_seconds = 0.0;
    double finstod_seconds = 0.0;
};

SmodTimingBreakdown compute_fresh_stod_finstod_slice(
    DynamicalSystem* sys,
    int rows, int cols,
    int start_row,
    int nx_global, int ny_global,
    double x_min, double x_max,
    double y_min, double y_max,
    double t_snap, double t_end,
    double dt,
    vector<double>& out_stod_score,
    vector<int>& out_stod_type,
    vector<double>& out_finstod_score,
    vector<int>& out_finstod_type,
    const string& direction_label,
    bool use_orthogonal = false
) {
    SmodTimingBreakdown timing_result;
    
    double integration_time = t_end - t_snap;
    if (abs(integration_time) < 1e-9) {
        fill(out_stod_score.begin(), out_stod_score.end(), 0.0);
        fill(out_stod_type.begin(), out_stod_type.end(), -1);
        fill(out_finstod_score.begin(), out_finstod_score.end(), 0.0);
        fill(out_finstod_type.begin(), out_finstod_type.end(), -1);
        return timing_result;
    }

    double direction = (integration_time > 0) ? 1.0 : -1.0;
    double dx = (x_max - x_min) / nx_global;
    double dy = (y_max - y_min) / ny_global;

    cout << "[INFO] Starting " << direction_label << " FRESH STOD/FINSTOD (from t=" 
         << t_snap << " to t=" << t_end << ")..." << endl;

    // =========================================================
    // PHASE 1: Generate fresh trajectories and discretize paths
    // =========================================================
    auto t_traj_start = chrono::high_resolution_clock::now();
    
    vector<vector<Point>> grid_paths(rows * cols);

    #pragma omp parallel
    {
        vector<double> init_state(sys->get_dim());
        vector<double> traj_cache;
        vector<double> temp_state(sys->get_dim());
        
        #pragma omp for schedule(dynamic)
        for (int r = 0; r < rows; ++r) {
            int r_global = start_row + r;
            double y0 = y_min + (r_global + 0.5) * dy;

            for (int c = 0; c < cols; ++c) {
                double x0 = x_min + (c + 0.5) * dx;
                
                // Get initial state at grid cell center
                if (!sys->get_initial_state(x0, y0, init_state)) {
                    grid_paths[r*cols + c].clear();
                    continue;
                }

                // Fresh integration from t_snap to t_end
                traj_cache.clear();
                integrate_path_adaptive(sys, init_state, t_snap, t_end, dt, direction, traj_cache, use_orthogonal);
                
                if (traj_cache.empty()) {
                    grid_paths[r*cols + c].clear();
                    continue;
                }

                // Project trajectory to grid and discretize
                vector<double> proj_path;
                size_t dim = sys->get_dim();
                size_t n_points = traj_cache.size() / dim;
                proj_path.reserve(n_points * 2);
                
                for (size_t i = 0; i < n_points; ++i) {
                    for (size_t k = 0; k < dim; ++k) {
                        temp_state[k] = traj_cache[i * dim + k];
                    }
                    double px, py;
                    sys->project_to_grid(temp_state, px, py);
                    proj_path.push_back(px);
                    proj_path.push_back(py);
                }
                
                grid_paths[r*cols + c] = discretize_path(proj_path, nx_global, ny_global, x_min, x_max, y_min, y_max);
                
                // Ensure first point is the starting cell (matching Python: path[0] = [r, c])
                if (!grid_paths[r*cols + c].empty()) {
                    grid_paths[r*cols + c][0] = {r_global, c};
                }
            }
        }
    }
    
    auto t_traj_end = chrono::high_resolution_clock::now();
    timing_result.trajectory_seconds = chrono::duration<double>(t_traj_end - t_traj_start).count();
    cout << "[TIMING] Trajectory generation + discretization: " << fixed << setprecision(2) 
         << timing_result.trajectory_seconds << "s" << endl;

    // =========================================================
    // PHASE 2: STOD Scoring (forward paths)
    // =========================================================
    auto t_stod_start = chrono::high_resolution_clock::now();
    
    #pragma omp parallel for collapse(2)
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const auto& pC = grid_paths[r*cols + c];
            
            if (pC.empty()) {
                out_stod_type[r*cols + c] = -1;
                out_stod_score[r*cols + c] = 0.0;
                continue;
            }

            int best_stod_type = -1; 
            double best_stod_score = 0.0;

            int dr[] = {-1, 1, 0, 0};
            int dc[] = {0, 0, -1, 1};

            for(int i=0; i<4; ++i) {
                int nr = r + dr[i];
                int nc = c + dc[i];
                
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
                    const auto& pN = grid_paths[nr*cols + nc];
                    if (pN.empty()) continue;

                    // STOD: forward paths
                    SmodResult res_stod = calculate_stod_pair(pC, pN);
                    if (res_stod.type_code > best_stod_type) {
                        best_stod_type = res_stod.type_code;
                        best_stod_score = res_stod.score;
                    } 
                    else if (res_stod.type_code == best_stod_type) {
                        if (res_stod.score > best_stod_score) {
                            best_stod_score = res_stod.score;
                        }
                    }
                }
            }
            out_stod_type[r * cols + c] = best_stod_type;
            out_stod_score[r * cols + c] = best_stod_score;
        }
    }
    
    auto t_stod_end = chrono::high_resolution_clock::now();
    timing_result.stod_seconds = chrono::duration<double>(t_stod_end - t_stod_start).count();
    cout << "[TIMING] STOD scoring: " << fixed << setprecision(2) 
         << timing_result.stod_seconds << "s" << endl;

    // =========================================================
    // PHASE 3: FINSTOD Scoring (reversed paths)
    // =========================================================
    auto t_finstod_start = chrono::high_resolution_clock::now();
    
    #pragma omp parallel for collapse(2)
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const auto& pC = grid_paths[r*cols + c];
            
            if (pC.empty()) {
                out_finstod_type[r*cols + c] = -1;
                out_finstod_score[r*cols + c] = 0.0;
                continue;
            }

            // Create reversed path for FINSTOD
            vector<Point> pC_rev = pC;
            reverse(pC_rev.begin(), pC_rev.end());

            int best_finstod_type = -1; 
            double best_finstod_score = 0.0;

            int dr[] = {-1, 1, 0, 0};
            int dc[] = {0, 0, -1, 1};

            for(int i=0; i<4; ++i) {
                int nr = r + dr[i];
                int nc = c + dc[i];
                
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
                    const auto& pN = grid_paths[nr*cols + nc];
                    if (pN.empty()) continue;

                    // FINSTOD: reversed paths
                    vector<Point> pN_rev = pN;
                    reverse(pN_rev.begin(), pN_rev.end());
                    SmodResult res_finstod = calculate_stod_pair(pC_rev, pN_rev);
                    if (res_finstod.type_code > best_finstod_type) {
                        best_finstod_type = res_finstod.type_code;
                        best_finstod_score = res_finstod.score;
                    } 
                    else if (res_finstod.type_code == best_finstod_type) {
                        if (res_finstod.score > best_finstod_score) {
                            best_finstod_score = res_finstod.score;
                        }
                    }
                }
            }
            out_finstod_type[r * cols + c] = best_finstod_type;
            out_finstod_score[r * cols + c] = best_finstod_score;
        }
    }
    
    auto t_finstod_end = chrono::high_resolution_clock::now();
    timing_result.finstod_seconds = chrono::duration<double>(t_finstod_end - t_finstod_start).count();
    cout << "[TIMING] FINSTOD scoring: " << fixed << setprecision(2) 
         << timing_result.finstod_seconds << "s" << endl;
    
    return timing_result;
}

// =========================================================
// 3. EXACT OLB HELPER (LB using STOD instead of FTLE)
// =========================================================
// Computes STOD at a specific time slice t, similar to how LB computes FTLE
double get_stod_at_time_slice(
    DynamicalSystem* sys,
    double x0, double y0,
    double t_start, double t_end,
    int nx_global, int ny_global,
    double x_min, double x_max,
    double y_min, double y_max,
    double dt,
    bool use_finstod,  // true for FINSTOD (reversed), false for STOD (forward)
    vector<double>& center_state,
    vector<double>& traj_cache,
    vector<double>& temp_state,
    bool use_orthogonal = false
) {
    double duration = t_end - t_start;
    if (abs(duration) < 1e-5) return 0.0;

    // Get initial state at grid cell center
    if (!sys->get_initial_state(x0, y0, center_state)) {
        return 0.0;
    }

    // Fresh integration from t_start to t_end
    double direction = (duration > 0) ? 1.0 : -1.0;
    traj_cache.clear();
    integrate_path_adaptive(sys, center_state, t_start, t_end, dt, direction, traj_cache, use_orthogonal);
    
    if (traj_cache.empty()) {
        return 0.0;
    }

    // Project trajectory to grid and discretize
    vector<double> proj_path;
    size_t dim = sys->get_dim();
    size_t n_points = traj_cache.size() / dim;
    proj_path.reserve(n_points * 2);
    
    for (size_t i = 0; i < n_points; ++i) {
        for (size_t k = 0; k < dim; ++k) {
            temp_state[k] = traj_cache[i * dim + k];
        }
        double px, py;
        sys->project_to_grid(temp_state, px, py);
        proj_path.push_back(px);
        proj_path.push_back(py);
    }
    
    vector<Point> center_path = discretize_path(proj_path, nx_global, ny_global, x_min, x_max, y_min, y_max);
    
    // Get grid cell indices for center (NO CLAMPING - consistent with discretize_path)
    int r_center = (int)((y0 - y_min) / ((y_max - y_min) / ny_global));
    int c_center = (int)((x0 - x_min) / ((x_max - x_min) / nx_global));
    
    // Ensure first point is the starting cell
    if (!center_path.empty()) {
        center_path[0] = {r_center, c_center};
    }
    
    if (center_path.empty()) {
        return 0.0;
    }

    // Compute STOD with 4 neighbors
    // For neighbors, we need to integrate from their positions
    double dx = (x_max - x_min) / nx_global;
    double dy = (y_max - y_min) / ny_global;
    
    int dr[] = {-1, 1, 0, 0};
    int dc[] = {0, 0, -1, 1};
    
    int best_type = -1;
    double best_score = 0.0;
    
    for (int i = 0; i < 4; ++i) {
        int nr = r_center + dr[i];
        int nc = c_center + dc[i];
        
        if (nr < 0 || nr >= ny_global || nc < 0 || nc >= nx_global) {
            continue;
        }
        
        // Get neighbor position
        double nx0 = x_min + (nc + 0.5) * dx;
        double ny0 = y_min + (nr + 0.5) * dy;
        
        // Get initial state for neighbor
        if (!sys->get_initial_state(nx0, ny0, temp_state)) {
            continue;
        }
        
        // Integrate neighbor trajectory
        vector<double> neighbor_traj_cache;
        integrate_path_adaptive(sys, temp_state, t_start, t_end, dt, direction, neighbor_traj_cache, use_orthogonal);
        
        if (neighbor_traj_cache.empty()) {
            continue;
        }
        
        // Project and discretize neighbor path
        vector<double> neighbor_proj_path;
        size_t n_neighbor_points = neighbor_traj_cache.size() / dim;
        neighbor_proj_path.reserve(n_neighbor_points * 2);
        
        for (size_t j = 0; j < n_neighbor_points; ++j) {
            for (size_t k = 0; k < dim; ++k) {
                temp_state[k] = neighbor_traj_cache[j * dim + k];
            }
            double px, py;
            sys->project_to_grid(temp_state, px, py);
            neighbor_proj_path.push_back(px);
            neighbor_proj_path.push_back(py);
        }
        
        vector<Point> neighbor_path = discretize_path(neighbor_proj_path, nx_global, ny_global, x_min, x_max, y_min, y_max);
        
        // Ensure first point is the starting cell
        if (!neighbor_path.empty()) {
            neighbor_path[0] = {nr, nc};
        }
        
        if (neighbor_path.empty()) {
            continue;
        }
        
        // Compute STOD pair
        SmodResult res;
        if (use_finstod) {
            // FINSTOD: reversed paths
            vector<Point> center_rev = center_path;
            reverse(center_rev.begin(), center_rev.end());
            vector<Point> neighbor_rev = neighbor_path;
            reverse(neighbor_rev.begin(), neighbor_rev.end());
            res = calculate_stod_pair(center_rev, neighbor_rev);
        } else {
            // STOD: forward paths
            res = calculate_stod_pair(center_path, neighbor_path);
        }
        
        if (res.type_code > best_type) {
            best_type = res.type_code;
            best_score = res.score;
        } else if (res.type_code == best_type) {
            if (res.score > best_score) {
                best_score = res.score;
            }
        }
    }
    
    return best_score;
}

void compute_exact_olb_slice(
    DynamicalSystem* sys,
    int rows, int cols,
    int start_row,
    int nx_global, int ny_global,
    double x_min, double x_max,
    double y_min, double y_max,
    double t_total,
    double dt,
    int N_slices,
    bool use_finstod,  // true for FINSTOD-based OLB, false for STOD-based
    vector<double>& out_olb_fxb,  // F × B (classical OLB)
    vector<double>& out_olb_fpb,  // F + B
    vector<double>& out_olb_fmb,  // F - B
    bool use_orthogonal = false
) {
    double dx = (x_max - x_min) / nx_global;
    double dy = (y_max - y_min) / ny_global;
    double dt_slice = t_total / N_slices;

    string stod_type = use_finstod ? "FINSTOD" : "STOD";
    cout << "[INFO] Starting EXACT OLB Calculation (" << stod_type << ", " << N_slices << " time slices)..." << endl;

    atomic<int> completed_rows(0);
    int report_interval = max(1, rows / 10);
    auto t_timer_start = chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        vector<double> center_state(sys->get_dim());
        vector<double> traj_cache;
        vector<double> temp_state(sys->get_dim());
        
        #pragma omp for schedule(dynamic)
        for (int r = 0; r < rows; ++r) {
            int r_global = start_row + r;
            double y0 = y_min + (r_global + 0.5) * dy;

            for (int c = 0; c < cols; ++c) {
                double x0 = x_min + (c + 0.5) * dx;
                double sum_olb_fxb = 0.0;
                double sum_olb_fpb = 0.0;
                double sum_olb_fmb = 0.0;
                bool valid_pixel = true;

                // Integral over time (matching LB structure)
                for (int n = 0; n <= N_slices; ++n) {
                    double t = n * dt_slice;
                    
                    // Backward STOD: from t to 0
                    double stod_bwd = get_stod_at_time_slice(
                        sys, x0, y0, t, 0.0, nx_global, ny_global,
                        x_min, x_max, y_min, y_max, dt, use_finstod,
                        center_state, traj_cache, temp_state, use_orthogonal
                    );
                    
                    // Forward STOD: from t to t_total
                    double stod_fwd = get_stod_at_time_slice(
                        sys, x0, y0, t, t_total, nx_global, ny_global,
                        x_min, x_max, y_min, y_max, dt, use_finstod,
                        center_state, traj_cache, temp_state, use_orthogonal
                    );

                    if (stod_bwd < 0 || stod_fwd < 0) { 
                        valid_pixel = false; 
                        break; 
                    }
                    
                    // Ensure non-negative for F×B calculation
                    double stod_bwd_safe = max(stod_bwd, 0.0);
                    double stod_fwd_safe = max(stod_fwd, 0.0);
                    
                    // DEBUG: Check if forward and backward STOD are equal (they shouldn't always be)
                    static int debug_count = 0;
                    if (debug_count < 10 && r == rows/2 && c == cols/2) {
                        cout << "[DEBUG OLB] At t=" << t << ", stod_fwd=" << stod_fwd_safe 
                             << ", stod_bwd=" << stod_bwd_safe << endl;
                        debug_count++;
                    }
                    
                    // F × B (classical OLB): sqrt(stod_bwd * stod_fwd)
                    double contrib_fxb = sqrt(stod_bwd_safe * stod_fwd_safe);
                    sum_olb_fxb += contrib_fxb;
                    
                    // F + B
                    double contrib_fpb = (stod_fwd_safe + stod_bwd_safe);
                    sum_olb_fpb += contrib_fpb;
                    
                    // F - B
                    sum_olb_fmb += (stod_fwd_safe - stod_bwd_safe);
                    
                    // DEBUG: Warn if contributions are equal when they shouldn't be
                    if (debug_count < 10 && r == rows/2 && c == cols/2 && 
                        abs(contrib_fxb - contrib_fpb) < 1e-10 && contrib_fxb > 1e-10) {
                        cout << "[DEBUG OLB] WARNING: fxb contribution (" << contrib_fxb 
                             << ") equals fpb contribution (" << contrib_fpb 
                             << ") at t=" << t << "!" << endl;
                    }
                }

                if (valid_pixel) {
                    double norm = dt_slice / t_total;
                    out_olb_fxb[r * cols + c] = sum_olb_fxb * norm;
                    out_olb_fpb[r * cols + c] = sum_olb_fpb * norm;
                    out_olb_fmb[r * cols + c] = sum_olb_fmb * norm;
                } else {
                    out_olb_fxb[r * cols + c] = 0.0;
                    out_olb_fpb[r * cols + c] = 0.0;
                    out_olb_fmb[r * cols + c] = 0.0;
                }
            }

            int cr = ++completed_rows;
            if (cr % report_interval == 0 || cr == rows) {
                #pragma omp critical
                {
                   auto t_now = chrono::high_resolution_clock::now();
                   double elapsed = chrono::duration<double>(t_now - t_timer_start).count();
                   double pct = (double)cr / rows * 100.0;
                   double eta = (elapsed / cr) * (rows - cr);
                   cout << "[PROGRESS] OLB Integral (" << stod_type << "): " << cr << "/" << rows 
                        << " (" << fixed << setprecision(1) << pct << "%) - ETA: " 
                        << format_duration(eta) << endl << flush;
                }
            }
        }
    }
}

// =========================================================
// 3b. CACHED OLB - MAJOR OPTIMIZATION
// =========================================================
// This version integrates each trajectory ONCE and extracts sub-segments
// for each time slice, reducing computational cost dramatically.
//
// For N_slices=300 and 4 neighbors:
//   OLD: ~3000 integrations per pixel
//   NEW: ~5 integrations per pixel (center + 4 neighbors, once each)
//
// The key insight is that the same trajectory from 0 to t_total can be
// sliced to get segments for any [t_start, t_end] within that range.

// Helper: Extract a portion of a discretized path corresponding to time range
// traj_times[i] = time at which traj point i was recorded
// Returns the discretized path segment from t_start to t_end
vector<Point> extract_path_segment(
    const vector<Point>& full_path,
    const vector<double>& traj_times,
    double t_start, double t_end,
    int start_cell_r, int start_cell_c
) {
    if (full_path.empty() || traj_times.empty()) return {};
    
    // Find indices corresponding to t_start and t_end
    size_t idx_start = 0;
    size_t idx_end = full_path.size() - 1;
    
    // Find start index (first point with time >= t_start)
    for (size_t i = 0; i < traj_times.size(); ++i) {
        if (traj_times[i] >= t_start - 1e-9) {
            idx_start = i;
            break;
        }
    }
    
    // Find end index (last point with time <= t_end)
    for (size_t i = traj_times.size(); i > 0; --i) {
        if (traj_times[i-1] <= t_end + 1e-9) {
            idx_end = i - 1;
            break;
        }
    }
    
    if (idx_start > idx_end) return {};
    
    vector<Point> segment;
    segment.reserve(idx_end - idx_start + 1);
    for (size_t i = idx_start; i <= idx_end; ++i) {
        segment.push_back(full_path[i]);
    }
    
    // Ensure first point is the correct starting cell
    if (!segment.empty()) {
        segment[0] = {start_cell_r, start_cell_c};
    }
    
    return segment;
}

// Compute STOD from pre-computed discretized paths (cached version)
double compute_stod_from_cached_paths(
    const vector<Point>& center_path,
    const vector<Point>& neighbor_path,
    bool use_finstod
) {
    if (center_path.empty() || neighbor_path.empty()) return 0.0;
    
    SmodResult res;
    if (use_finstod) {
        vector<Point> center_rev = center_path;
        reverse(center_rev.begin(), center_rev.end());
        vector<Point> neighbor_rev = neighbor_path;
        reverse(neighbor_rev.begin(), neighbor_rev.end());
        res = calculate_stod_pair(center_rev, neighbor_rev);
    } else {
        res = calculate_stod_pair(center_path, neighbor_path);
    }
    return res.score;
}

// Structure to hold cached trajectory data
struct CachedTrajectory {
    vector<Point> discretized_path;  // Full discretized path from 0 to t_total
    vector<double> times;            // Time at each point
    int start_r, start_c;            // Starting grid cell
    bool valid;
};

// Integrate and cache a single trajectory with time stamps
CachedTrajectory integrate_and_cache_trajectory(
    DynamicalSystem* sys,
    double x0, double y0,
    double t_total, double dt,
    int nx_global, int ny_global,
    double x_min, double x_max,
    double y_min, double y_max,
    bool use_orthogonal
) {
    CachedTrajectory result;
    result.valid = false;
    
    size_t dim = sys->get_dim();
    vector<double> state(dim);
    
    if (!sys->get_initial_state(x0, y0, state)) {
        return result;
    }
    
    // Integrate full trajectory from 0 to t_total
    vector<double> traj_cache;
    integrate_path_adaptive(sys, state, 0.0, t_total, dt, 1.0, traj_cache, use_orthogonal);
    
    if (traj_cache.empty()) {
        return result;
    }
    
    // Project to grid and record times
    size_t n_points = traj_cache.size() / dim;
    vector<double> proj_path;
    proj_path.reserve(n_points * 2);
    result.times.reserve(n_points);
    
    vector<double> temp_state(dim);
    for (size_t i = 0; i < n_points; ++i) {
        for (size_t k = 0; k < dim; ++k) {
            temp_state[k] = traj_cache[i * dim + k];
        }
        double px, py;
        sys->project_to_grid(temp_state, px, py);
        proj_path.push_back(px);
        proj_path.push_back(py);
        
        // Time at this point (assuming uniform output spacing from integrator)
        result.times.push_back(i * dt);
    }
    
    // Discretize to grid cells
    result.discretized_path = discretize_path(proj_path, nx_global, ny_global, x_min, x_max, y_min, y_max);
    
    // Match times to discretized path (may have different length due to cell merging)
    // For simplicity, we'll interpolate times
    if (result.discretized_path.size() != result.times.size() && !result.discretized_path.empty()) {
        // Resample times to match discretized path length
        vector<double> new_times;
        new_times.reserve(result.discretized_path.size());
        double time_step = t_total / (result.discretized_path.size() - 1);
        for (size_t i = 0; i < result.discretized_path.size(); ++i) {
            new_times.push_back(i * time_step);
        }
        result.times = new_times;
    }
    
    // Store starting cell
    double cell_dx = (x_max - x_min) / nx_global;
    double cell_dy = (y_max - y_min) / ny_global;
    result.start_r = (int)((y0 - y_min) / cell_dy);
    result.start_c = (int)((x0 - x_min) / cell_dx);
    
    // Ensure first point is starting cell
    if (!result.discretized_path.empty()) {
        result.discretized_path[0] = {result.start_r, result.start_c};
    }
    
    result.valid = !result.discretized_path.empty();
    return result;
}

// =========================================================
// CORRECT CACHED OLB FOR AUTONOMOUS SYSTEMS
// =========================================================
// For autonomous systems, STOD(x0, t1->t2) depends only on duration |t2-t1|.
// Therefore:
//   - STOD_fwd at time slice t (from t to τ) = STOD over duration (τ-t)
//   - STOD_bwd at time slice t (from t to 0) = STOD over duration t (reversed)
//
// We integrate once from 0 to τ, then extract segments:
//   - Forward at slice t: segment [0, τ-t]
//   - Backward at slice t: segment [0, t] reversed
//
// This gives ~N_slices × speedup compared to the non-cached version.

void compute_exact_olb_slice_cached_v2(
    DynamicalSystem* sys,
    int rows, int cols,
    int start_row,
    int nx_global, int ny_global,
    double x_min, double x_max,
    double y_min, double y_max,
    double t_total,
    double dt,
    int N_slices,
    bool use_finstod,
    vector<double>& out_olb_fxb,
    vector<double>& out_olb_fpb,
    vector<double>& out_olb_fmb,
    bool use_orthogonal = false
) {
    double dx = (x_max - x_min) / nx_global;
    double dy = (y_max - y_min) / ny_global;
    double dt_slice = t_total / N_slices;

    string stod_type = use_finstod ? "FINSTOD" : "STOD";
    cout << "[INFO] Starting CACHED OLB v2 (" << stod_type << ", " << N_slices << " time slices)..." << endl;
    cout << "[INFO] Using autonomous-system caching for ~" << N_slices << "x speedup!" << endl;

    atomic<int> completed_rows(0);
    int report_interval = max(1, rows / 10);
    auto t_timer_start = chrono::high_resolution_clock::now();

    int dr[] = {-1, 1, 0, 0};
    int dc[] = {0, 0, -1, 1};

    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for (int r = 0; r < rows; ++r) {
            int r_global = start_row + r;
            double y0 = y_min + (r_global + 0.5) * dy;

            for (int c = 0; c < cols; ++c) {
                double x0 = x_min + (c + 0.5) * dx;
                
                // Step 1: Integrate center trajectory ONCE from 0 to t_total
                CachedTrajectory center_traj = integrate_and_cache_trajectory(
                    sys, x0, y0, t_total, dt, nx_global, ny_global,
                    x_min, x_max, y_min, y_max, use_orthogonal
                );
                
                if (!center_traj.valid) {
                    out_olb_fxb[r * cols + c] = 0.0;
                    out_olb_fpb[r * cols + c] = 0.0;
                    out_olb_fmb[r * cols + c] = 0.0;
                    continue;
                }
                
                // Step 2: Integrate neighbor trajectories ONCE each
                vector<CachedTrajectory> neighbor_trajs(4);
                int start_cell_r = center_traj.start_r;
                int start_cell_c = center_traj.start_c;
                
                for (int i = 0; i < 4; ++i) {
                    int nr = start_cell_r + dr[i];
                    int nc = start_cell_c + dc[i];
                    
                    if (nr < 0 || nr >= ny_global || nc < 0 || nc >= nx_global) {
                        neighbor_trajs[i].valid = false;
                        continue;
                    }
                    
                    double nx0 = x_min + (nc + 0.5) * dx;
                    double ny0 = y_min + (nr + 0.5) * dy;
                    
                    neighbor_trajs[i] = integrate_and_cache_trajectory(
                        sys, nx0, ny0, t_total, dt, nx_global, ny_global,
                        x_min, x_max, y_min, y_max, use_orthogonal
                    );
                }
                
                // Step 3: Loop over time slices, using segments of appropriate duration
                double sum_olb_fxb = 0.0;
                double sum_olb_fpb = 0.0;
                double sum_olb_fmb = 0.0;
                bool valid_pixel = true;

                for (int n = 0; n <= N_slices && valid_pixel; ++n) {
                    double t = n * dt_slice;
                    double duration_fwd = t_total - t;  // Duration for forward STOD
                    double duration_bwd = t;             // Duration for backward STOD
                    
                    // Extract center path segments based on DURATION (not position along trajectory)
                    // For autonomous systems: STOD over duration D starting from x0 
                    // = segment [0, D] of trajectory starting from x0
                    
                    // Forward: need segment of duration (τ - t), i.e., [0, τ-t]
                    vector<Point> center_fwd_segment;
                    if (duration_fwd > 1e-9) {
                        center_fwd_segment = extract_path_segment(
                            center_traj.discretized_path, center_traj.times,
                            0.0, duration_fwd, center_traj.start_r, center_traj.start_c
                        );
                    }
                    
                    // Backward: need segment of duration t, i.e., [0, t], then reversed
                    vector<Point> center_bwd_segment;
                    if (duration_bwd > 1e-9) {
                        center_bwd_segment = extract_path_segment(
                            center_traj.discretized_path, center_traj.times,
                            0.0, duration_bwd, center_traj.start_r, center_traj.start_c
                        );
                        // Reverse for backward interpretation
                        if (!center_bwd_segment.empty()) {
                            reverse(center_bwd_segment.begin(), center_bwd_segment.end());
                        }
                    }
                    
                    // Compute STOD for each neighbor and take best
                    double stod_bwd = 0.0;
                    double stod_fwd = 0.0;
                    
                    for (int i = 0; i < 4; ++i) {
                        if (!neighbor_trajs[i].valid) continue;
                        
                        // Extract neighbor segments with same durations
                        vector<Point> neighbor_fwd_segment;
                        if (duration_fwd > 1e-9) {
                            neighbor_fwd_segment = extract_path_segment(
                                neighbor_trajs[i].discretized_path, neighbor_trajs[i].times,
                                0.0, duration_fwd, neighbor_trajs[i].start_r, neighbor_trajs[i].start_c
                            );
                        }
                        
                        vector<Point> neighbor_bwd_segment;
                        if (duration_bwd > 1e-9) {
                            neighbor_bwd_segment = extract_path_segment(
                                neighbor_trajs[i].discretized_path, neighbor_trajs[i].times,
                                0.0, duration_bwd, neighbor_trajs[i].start_r, neighbor_trajs[i].start_c
                            );
                            if (!neighbor_bwd_segment.empty()) {
                                reverse(neighbor_bwd_segment.begin(), neighbor_bwd_segment.end());
                            }
                        }
                        
                        // Compute STOD for this neighbor pair
                        if (!center_bwd_segment.empty() && !neighbor_bwd_segment.empty()) {
                            double score = compute_stod_from_cached_paths(
                                center_bwd_segment, neighbor_bwd_segment, use_finstod
                            );
                            if (score > stod_bwd) stod_bwd = score;
                        }
                        
                        if (!center_fwd_segment.empty() && !neighbor_fwd_segment.empty()) {
                            double score = compute_stod_from_cached_paths(
                                center_fwd_segment, neighbor_fwd_segment, use_finstod
                            );
                            if (score > stod_fwd) stod_fwd = score;
                        }
                    }
                    
                    // Accumulate OLB contributions
                    double stod_bwd_safe = max(stod_bwd, 0.0);
                    double stod_fwd_safe = max(stod_fwd, 0.0);
                    
                    // F × B
                    sum_olb_fxb += sqrt(stod_bwd_safe * stod_fwd_safe);
                    // F + B
                    sum_olb_fpb += (stod_fwd_safe + stod_bwd_safe);
                    // F - B
                    sum_olb_fmb += (stod_fwd_safe - stod_bwd_safe);
                }

                if (valid_pixel) {
                    double norm = dt_slice / t_total;
                    out_olb_fxb[r * cols + c] = sum_olb_fxb * norm;
                    out_olb_fpb[r * cols + c] = sum_olb_fpb * norm;
                    out_olb_fmb[r * cols + c] = sum_olb_fmb * norm;
                } else {
                    out_olb_fxb[r * cols + c] = 0.0;
                    out_olb_fpb[r * cols + c] = 0.0;
                    out_olb_fmb[r * cols + c] = 0.0;
                }
            }

            int cr = ++completed_rows;
            if (cr % report_interval == 0 || cr == rows) {
                #pragma omp critical
                {
                   auto t_now = chrono::high_resolution_clock::now();
                   double elapsed = chrono::duration<double>(t_now - t_timer_start).count();
                   double pct = (double)cr / rows * 100.0;
                   double eta = (elapsed / cr) * (rows - cr);
                   cout << "[PROGRESS] OLB Integral (" << stod_type << "): " << cr << "/" << rows 
                        << " (" << fixed << setprecision(1) << pct << "%) - ETA: " 
                        << format_duration(eta) << endl;
                }
            }
        }
    }
}

// OLD CACHED VERSION (INCORRECT - kept for reference)
// This version tracked where the particle actually went, which is NOT
// what OLB requires. OLB needs STOD computed from the same initial point
// but with different integration durations.
void compute_exact_olb_slice_cached(
    DynamicalSystem* sys,
    int rows, int cols,
    int start_row,
    int nx_global, int ny_global,
    double x_min, double x_max,
    double y_min, double y_max,
    double t_total,
    double dt,
    int N_slices,
    bool use_finstod,
    vector<double>& out_olb_fxb,
    vector<double>& out_olb_fpb,
    vector<double>& out_olb_fmb,
    bool use_orthogonal = false
) {
    double dx = (x_max - x_min) / nx_global;
    double dy = (y_max - y_min) / ny_global;
    double dt_slice = t_total / N_slices;

    string stod_type = use_finstod ? "FINSTOD" : "STOD";
    cout << "[INFO] Starting CACHED OLB Calculation (" << stod_type << ", " << N_slices << " time slices)..." << endl;
    cout << "[INFO] Using trajectory caching for ~" << (N_slices * 10) << "x speedup!" << endl;

    atomic<int> completed_rows(0);
    int report_interval = max(1, rows / 10);
    auto t_timer_start = chrono::high_resolution_clock::now();

    int dr[] = {-1, 1, 0, 0};
    int dc[] = {0, 0, -1, 1};

    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for (int r = 0; r < rows; ++r) {
            int r_global = start_row + r;
            double y0 = y_min + (r_global + 0.5) * dy;

            for (int c = 0; c < cols; ++c) {
                double x0 = x_min + (c + 0.5) * dx;
                
                // Step 1: Integrate center trajectory ONCE from 0 to t_total
                CachedTrajectory center_traj = integrate_and_cache_trajectory(
                    sys, x0, y0, t_total, dt, nx_global, ny_global,
                    x_min, x_max, y_min, y_max, use_orthogonal
                );
                
                if (!center_traj.valid) {
                    out_olb_fxb[r * cols + c] = 0.0;
                    out_olb_fpb[r * cols + c] = 0.0;
                    out_olb_fmb[r * cols + c] = 0.0;
                    continue;
                }
                
                // Step 2: Integrate neighbor trajectories ONCE each
                vector<CachedTrajectory> neighbor_trajs(4);
                for (int i = 0; i < 4; ++i) {
                    int nr = center_traj.start_r + dr[i];
                    int nc = center_traj.start_c + dc[i];
                    
                    if (nr < 0 || nr >= ny_global || nc < 0 || nc >= nx_global) {
                        neighbor_trajs[i].valid = false;
                        continue;
                    }
                    
                    double nx0 = x_min + (nc + 0.5) * dx;
                    double ny0 = y_min + (nr + 0.5) * dy;
                    
                    neighbor_trajs[i] = integrate_and_cache_trajectory(
                        sys, nx0, ny0, t_total, dt, nx_global, ny_global,
                        x_min, x_max, y_min, y_max, use_orthogonal
                    );
                }
                
                // Step 3: Loop over time slices, extracting segments from cached trajectories
                double sum_olb_fxb = 0.0;
                double sum_olb_fpb = 0.0;
                double sum_olb_fmb = 0.0;
                bool valid_pixel = true;

                for (int n = 0; n <= N_slices && valid_pixel; ++n) {
                    double t = n * dt_slice;
                    
                    // Extract center path segments
                    // Backward: from position at t=0, looking at segment [0, t] reversed
                    // Forward: from position at t=0, looking at segment [t, t_total]
                    
                    // For backward STOD (measuring "where did I come from"):
                    // We need the path segment from t to 0 (i.e., [0, t] reversed)
                    vector<Point> center_bwd_segment = extract_path_segment(
                        center_traj.discretized_path, center_traj.times,
                        0.0, t, center_traj.start_r, center_traj.start_c
                    );
                    // Reverse it for backward interpretation
                    if (!center_bwd_segment.empty()) {
                        reverse(center_bwd_segment.begin(), center_bwd_segment.end());
                    }
                    
                    // For forward STOD (measuring "where will I go"):
                    // We need the path segment from t to t_total
                    vector<Point> center_fwd_segment = extract_path_segment(
                        center_traj.discretized_path, center_traj.times,
                        t, t_total, center_traj.start_r, center_traj.start_c
                    );
                    
                    if (center_bwd_segment.empty() && t > 1e-9) {
                        valid_pixel = false;
                        break;
                    }
                    if (center_fwd_segment.empty() && (t_total - t) > 1e-9) {
                        valid_pixel = false;
                        break;
                    }
                    
                    // Compute STOD for each neighbor and take best
                    double stod_bwd = 0.0;
                    double stod_fwd = 0.0;
                    
                    for (int i = 0; i < 4; ++i) {
                        if (!neighbor_trajs[i].valid) continue;
                        
                        // Extract neighbor segments
                        vector<Point> neighbor_bwd_segment = extract_path_segment(
                            neighbor_trajs[i].discretized_path, neighbor_trajs[i].times,
                            0.0, t, neighbor_trajs[i].start_r, neighbor_trajs[i].start_c
                        );
                        if (!neighbor_bwd_segment.empty()) {
                            reverse(neighbor_bwd_segment.begin(), neighbor_bwd_segment.end());
                        }
                        
                        vector<Point> neighbor_fwd_segment = extract_path_segment(
                            neighbor_trajs[i].discretized_path, neighbor_trajs[i].times,
                            t, t_total, neighbor_trajs[i].start_r, neighbor_trajs[i].start_c
                        );
                        
                        // Compute STOD for this neighbor pair
                        if (!center_bwd_segment.empty() && !neighbor_bwd_segment.empty()) {
                            double score = compute_stod_from_cached_paths(
                                center_bwd_segment, neighbor_bwd_segment, use_finstod
                            );
                            if (score > stod_bwd) stod_bwd = score;
                        }
                        
                        if (!center_fwd_segment.empty() && !neighbor_fwd_segment.empty()) {
                            double score = compute_stod_from_cached_paths(
                                center_fwd_segment, neighbor_fwd_segment, use_finstod
                            );
                            if (score > stod_fwd) stod_fwd = score;
                        }
                    }
                    
                    // Accumulate OLB contributions
                    double stod_bwd_safe = max(stod_bwd, 0.0);
                    double stod_fwd_safe = max(stod_fwd, 0.0);
                    
                    sum_olb_fxb += sqrt(stod_bwd_safe * stod_fwd_safe);
                    sum_olb_fpb += (stod_fwd_safe + stod_bwd_safe);
                    sum_olb_fmb += (stod_fwd_safe - stod_bwd_safe);
                }

                if (valid_pixel) {
                    double norm = dt_slice / t_total;
                    out_olb_fxb[r * cols + c] = sum_olb_fxb * norm;
                    out_olb_fpb[r * cols + c] = sum_olb_fpb * norm;
                    out_olb_fmb[r * cols + c] = sum_olb_fmb * norm;
                } else {
                    out_olb_fxb[r * cols + c] = 0.0;
                    out_olb_fpb[r * cols + c] = 0.0;
                    out_olb_fmb[r * cols + c] = 0.0;
                }
            }

            int cr = ++completed_rows;
            if (cr % report_interval == 0 || cr == rows) {
                #pragma omp critical
                {
                   auto t_now = chrono::high_resolution_clock::now();
                   double elapsed = chrono::duration<double>(t_now - t_timer_start).count();
                   double pct = (double)cr / rows * 100.0;
                   double eta = (elapsed / cr) * (rows - cr);
                   cout << "[PROGRESS] Cached OLB Integral (" << stod_type << "): " << cr << "/" << rows 
                        << " (" << fixed << setprecision(1) << pct << "%) - ETA: " 
                        << format_duration(eta) << endl << flush;
                }
            }
        }
    }
}

// =========================================================
// 4. EXACT LB HELPER (From Pipeline 2)
// =========================================================
double get_lambda_max(
    DynamicalSystem* sys,
    double x0, double y0,
    double t_start, double t_end,
    vector<double>& center_state,
    vector<double>& traj_cache,
    bool use_orthogonal = false
) {
    double duration = t_end - t_start;
    if (abs(duration) < 1e-5) return 1.0;

    double direction = (duration > 0) ? 1.0 : -1.0;
    double eps = 1e-4; 

    double neighbors[4][2] = {
        {x0 + eps, y0}, {x0 - eps, y0}, 
        {x0, y0 + eps}, {x0, y0 - eps} 
    };
    double final_pos[4][2];

    for(int i=0; i<4; ++i) {
        if(!sys->get_initial_state(neighbors[i][0], neighbors[i][1], center_state)) {
            return -1.0; 
        }
        traj_cache.clear();
        integrate_path_adaptive(sys, center_state, t_start, t_end, abs(duration), direction, traj_cache, use_orthogonal);
        
        if(traj_cache.empty()) return -1.0;

        size_t dim = sys->get_dim();
        size_t last_idx = traj_cache.size() - dim;
        vector<double> final_full(dim);
        for(size_t k=0; k<dim; ++k) final_full[k] = traj_cache[last_idx + k];
        
        sys->project_to_grid(final_full, final_pos[i][0], final_pos[i][1]);
    }

    double dfx_dx = (final_pos[0][0] - final_pos[1][0]) / (2.0 * eps);
    double dfy_dx = (final_pos[0][1] - final_pos[1][1]) / (2.0 * eps);
    double dfx_dy = (final_pos[2][0] - final_pos[3][0]) / (2.0 * eps);
    double dfy_dy = (final_pos[2][1] - final_pos[3][1]) / (2.0 * eps);

    double c11 = dfx_dx*dfx_dx + dfy_dx*dfy_dx;
    double c12 = dfx_dx*dfx_dy + dfy_dx*dfy_dy; 
    double c22 = dfx_dy*dfx_dy + dfy_dy*dfy_dy;

    double tr = c11 + c22;
    double det = c11*c22 - c12*c12;
    double gap = sqrt(max(0.0, tr*tr - 4*det));
    return (tr + gap) / 2.0; 
}

void compute_exact_lb_slice(
    DynamicalSystem* sys,
    int rows, int cols,
    int start_row,
    int nx_global, int ny_global,
    double x_min, double x_max,
    double y_min, double y_max,
    double t_total, 
    int N_slices,
    vector<double>& out_lb_fxb,  // F × B (classical LB)
    vector<double>& out_lb_fpb,  // F + B
    vector<double>& out_lb_fmb,  // F - B
    bool use_orthogonal = false
) {
    double dx = (x_max - x_min) / nx_global;
    double dy = (y_max - y_min) / ny_global;
    double dt_slice = t_total / N_slices;

    cout << "[INFO] Starting EXACT LB Calculation (" << N_slices << " time slices)..." << endl;

    atomic<int> completed_rows(0);
    int report_interval = max(1, rows / 10);
    auto t_timer_start = chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        vector<double> center_state(sys->get_dim());
        vector<double> traj_cache; 
        
        #pragma omp for schedule(dynamic)
        for (int r = 0; r < rows; ++r) {
            int r_global = start_row + r;
            double y0 = y_min + (r_global + 0.5) * dy;

            for (int c = 0; c < cols; ++c) {
                double x0 = x_min + (c + 0.5) * dx;
                double sum_lb_fxb = 0.0;
                double sum_lb_fpb = 0.0;
                double sum_lb_fmb = 0.0;
                bool valid_pixel = true;

                // Integral over time
                for (int n = 0; n <= N_slices; ++n) {
                    double t = n * dt_slice;
                    double lam_bwd = get_lambda_max(sys, x0, y0, t, 0.0, center_state, traj_cache, use_orthogonal);
                    double lam_fwd = get_lambda_max(sys, x0, y0, t, t_total, center_state, traj_cache, use_orthogonal);

                    if (lam_bwd < 0 || lam_fwd < 0) { valid_pixel = false; break; }
                    
                    // F × B (classical LB): sqrt(lam_bwd) * sqrt(lam_fwd) = sqrt(lam_bwd * lam_fwd)
                    sum_lb_fxb += (sqrt(lam_bwd) * sqrt(lam_fwd));
                    
                    // F + B
                    sum_lb_fpb += (lam_fwd + lam_bwd);
                    
                    // F - B
                    sum_lb_fmb += (lam_fwd - lam_bwd);
                }

                if (valid_pixel) {
                    double norm = dt_slice / t_total;
                    out_lb_fxb[r * cols + c] = sum_lb_fxb * norm;
                    out_lb_fpb[r * cols + c] = sum_lb_fpb * norm;
                    out_lb_fmb[r * cols + c] = sum_lb_fmb * norm;
                } else {
                    out_lb_fxb[r * cols + c] = 0.0;
                    out_lb_fpb[r * cols + c] = 0.0;
                    out_lb_fmb[r * cols + c] = 0.0;
                }
            }

            int cr = ++completed_rows;
            if (cr % report_interval == 0 || cr == rows) {
                #pragma omp critical
                {
                   auto t_now = chrono::high_resolution_clock::now();
                   double elapsed = chrono::duration<double>(t_now - t_timer_start).count();
                   double pct = (double)cr / rows * 100.0;
                   double eta = (elapsed / cr) * (rows - cr);
                   cout << "[PROGRESS] LB Integral: " << cr << "/" << rows 
                        << " (" << fixed << setprecision(1) << pct << "%) - ETA: " 
                        << format_duration(eta) << endl << flush;
                }
            }
        }
    }
}

// =========================================================
// 5. FLI (Fast Lyapunov Indicator) CALCULATION
// =========================================================
// Uses the variational equations to compute the maximum stretching factor.
// FLI = log(max column norm of the tangent matrix Phi)
void compute_fli_slice(
    DynamicalSystem* sys,
    int rows, int cols,
    int start_row,
    int nx_global, int ny_global,
    double x_min, double x_max,
    double y_min, double y_max,
    double t_start, double t_end,
    double alpha_rescale,
    vector<double>& out_fli,
    const string& label,
    bool use_orthogonal = false
) {
    double integration_time = t_end - t_start;
    if (abs(integration_time) < 1e-9) {
        fill(out_fli.begin(), out_fli.end(), 0.0);
        return;
    }

    double dx = (x_max - x_min) / nx_global;
    double dy = (y_max - y_min) / ny_global;

    cout << "[INFO] Starting " << label << " FLI (variational) (" 
         << t_start << " -> " << t_end << ")..." << endl;

    atomic<int> completed_rows(0);
    int report_interval = max(1, rows / 10);
    auto t_timer_start = chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        vector<double> init_state(sys->get_dim());
        
        #pragma omp for schedule(dynamic)
        for (int r = 0; r < rows; ++r) {
            int r_global = start_row + r;
            double y0 = y_min + (r_global + 0.5) * dy;

            for (int c = 0; c < cols; ++c) {
                double x0 = x_min + (c + 0.5) * dx;
                
                if (!sys->get_initial_state(x0, y0, init_state)) {
                    out_fli[r * cols + c] = 0.0;
                    continue;
                }

                VariationalResult res = integrate_variational(
                    sys, init_state, t_start, t_end, alpha_rescale, 0.5, use_orthogonal
                );
                
                // Normalize FLI by integration time
                out_fli[r * cols + c] = res.fli_value / abs(integration_time);
            }

            // Progress reporting
            int cr = ++completed_rows;
            if (cr % report_interval == 0 || cr == rows) {
                #pragma omp critical
                {
                   auto t_now = chrono::high_resolution_clock::now();
                   double elapsed = chrono::duration<double>(t_now - t_timer_start).count();
                   double pct = (double)cr / rows * 100.0;
                   double eta = (elapsed / cr) * (rows - cr);
                   cout << "[PROGRESS] " << label << " FLI: " << cr << "/" << rows 
                        << " (" << fixed << setprecision(1) << pct << "%) - ETA: " 
                        << format_duration(eta) << endl << flush;
                }
            }
        }
    }
}

// =========================================================
// 6. LD (Lagrangian Descriptor) CALCULATION
// =========================================================
// Computes the integral of the p-norm of the velocity along trajectories.
// LD(x0, t0, tau) = integral from t0-tau to t0+tau of ||v(t)||_p dt
// Now outputs forward and backward separately for superimposition support.
void compute_ld_slice(
    DynamicalSystem* sys,
    int rows, int cols,
    int start_row,
    int nx_global, int ny_global,
    double x_min, double x_max,
    double y_min, double y_max,
    double t_snap, double t_total,
    double ld_p_norm,
    bool compute_forward,
    bool compute_backward,
    vector<double>& out_ld_fwd,
    vector<double>& out_ld_bwd,
    const string& label,
    bool use_orthogonal = false
) {
    double dx = (x_max - x_min) / nx_global;
    double dy = (y_max - y_min) / ny_global;

    string direction_str = "";
    if (compute_forward && compute_backward) direction_str = "Forward+Backward";
    else if (compute_forward) direction_str = "Forward";
    else if (compute_backward) direction_str = "Backward";

    cout << "[INFO] Starting " << label << " LD " << direction_str << " (p=" << ld_p_norm << ", t_snap=" 
         << t_snap << ", t_total=" << t_total << ")..." << endl;

    atomic<int> completed_rows(0);
    int report_interval = max(1, rows / 10);
    auto t_timer_start = chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        vector<double> init_state(sys->get_dim());
        
        #pragma omp for schedule(dynamic)
        for (int r = 0; r < rows; ++r) {
            int r_global = start_row + r;
            double y0 = y_min + (r_global + 0.5) * dy;

            for (int c = 0; c < cols; ++c) {
                double x0 = x_min + (c + 0.5) * dx;
                
                if (!sys->get_initial_state(x0, y0, init_state)) {
                    if (compute_forward) out_ld_fwd[r * cols + c] = 0.0;
                    if (compute_backward) out_ld_bwd[r * cols + c] = 0.0;
                    continue;
                }

                // Forward integration: from t_snap to t_total
                // Using simple LD integrator (no variational equations needed!)
                // This is much faster: dim ODEs instead of dim + dim^2 ODEs
                if (compute_forward && (t_total - t_snap > 1e-9)) {
                    LDResult res_fwd = integrate_path_with_ld(
                        sys, init_state, t_snap, t_total, ld_p_norm, use_orthogonal
                    );
                    out_ld_fwd[r * cols + c] = res_fwd.ld_value;
                } else if (compute_forward) {
                    out_ld_fwd[r * cols + c] = 0.0;
                }
                
                // Backward integration: from t_snap to 0
                if (compute_backward && (t_snap > 1e-9)) {
                    LDResult res_bwd = integrate_path_with_ld(
                        sys, init_state, t_snap, 0.0, ld_p_norm, use_orthogonal
                    );
                    out_ld_bwd[r * cols + c] = res_bwd.ld_value;
                } else if (compute_backward) {
                    out_ld_bwd[r * cols + c] = 0.0;
                }
            }

            // Progress reporting
            int cr = ++completed_rows;
            if (cr % report_interval == 0 || cr == rows) {
                #pragma omp critical
                {
                   auto t_now = chrono::high_resolution_clock::now();
                   double elapsed = chrono::duration<double>(t_now - t_timer_start).count();
                   double pct = (double)cr / rows * 100.0;
                   double eta = (elapsed / cr) * (rows - cr);
                   cout << "[PROGRESS] " << label << " LD: " << cr << "/" << rows 
                        << " (" << fixed << setprecision(1) << pct << "%) - ETA: " 
                        << format_duration(eta) << endl << flush;
                }
            }
        }
    }
}

// =========================================================
// 7. TIMING STATISTICS STRUCTURE
// =========================================================
struct TimingStats {
    // Standard metrics (each includes its own trajectory integration)
    double ftle_forward_seconds = 0.0;
    double ftle_backward_seconds = 0.0;
    double fli_forward_seconds = 0.0;
    double fli_backward_seconds = 0.0;
    double ld_forward_seconds = 0.0;
    double ld_backward_seconds = 0.0;
    
    // STOD metrics - broken down by phase
    double stod_traj_forward_seconds = 0.0;   // Trajectory generation + discretization (forward)
    double stod_traj_backward_seconds = 0.0;  // Trajectory generation + discretization (backward)
    double stod_forward_seconds = 0.0;      // STOD scoring only (forward)
    double stod_backward_seconds = 0.0;     // STOD scoring only (backward)
    double finstod_forward_seconds = 0.0;     // FINSTOD scoring only (forward)
    double finstod_backward_seconds = 0.0;    // FINSTOD scoring only (backward)
    
    void print_summary() const {
        double total_ftle = ftle_forward_seconds + ftle_backward_seconds;
        double total_fli = fli_forward_seconds + fli_backward_seconds;
        double total_ld = ld_forward_seconds + ld_backward_seconds;
        double total_standard = total_ftle + total_fli + total_ld;
        
        double total_stod_traj = stod_traj_forward_seconds + stod_traj_backward_seconds;
        double total_stod = stod_forward_seconds + stod_backward_seconds;
        double total_finstod = finstod_forward_seconds + finstod_backward_seconds;
        double total_stod = total_stod_traj + total_stod + total_finstod;
        
        double total_all = total_standard + total_stod;
        
        cout << endl;
        cout << "╔══════════════════════════════════════════════════════════════════════════════╗" << endl;
        cout << "║                         TIMING SUMMARY (this worker)                         ║" << endl;
        cout << "╠══════════════════════════════════════════════════════════════════════════════╣" << endl;
        cout << "║  STANDARD METRICS (each includes trajectory integration):                    ║" << endl;
        if (ftle_forward_seconds > 0 || ftle_backward_seconds > 0) {
            cout << "║    FTLE:     Fwd " << fixed << setprecision(2) << setw(7) << ftle_forward_seconds 
                 << "s | Bwd " << setw(7) << ftle_backward_seconds << "s | Total " 
                 << setw(7) << total_ftle << "s         ║" << endl;
        }
        if (fli_forward_seconds > 0 || fli_backward_seconds > 0) {
            cout << "║    FLI:      Fwd " << fixed << setprecision(2) << setw(7) << fli_forward_seconds 
                 << "s | Bwd " << setw(7) << fli_backward_seconds << "s | Total " 
                 << setw(7) << total_fli << "s         ║" << endl;
        }
        if (ld_forward_seconds > 0 || ld_backward_seconds > 0) {
            cout << "║    LD:       Fwd " << fixed << setprecision(2) << setw(7) << ld_forward_seconds 
                 << "s | Bwd " << setw(7) << ld_backward_seconds << "s | Total " 
                 << setw(7) << total_ld << "s         ║" << endl;
        }
        cout << "║    ─────────────────────────────────────────────────────────────────────     ║" << endl;
        cout << "║    Standard Total:                                       " 
             << setw(7) << total_standard << "s         ║" << endl;
        cout << "╠══════════════════════════════════════════════════════════════════════════════╣" << endl;
        cout << "║  STOD METRICS (broken down by phase):                                        ║" << endl;
        if (stod_traj_forward_seconds > 0 || stod_traj_backward_seconds > 0) {
            cout << "║    Traj Gen: Fwd " << fixed << setprecision(2) << setw(7) << stod_traj_forward_seconds 
                 << "s | Bwd " << setw(7) << stod_traj_backward_seconds << "s | Total " 
                 << setw(7) << total_stod_traj << "s         ║" << endl;
        }
        if (stod_forward_seconds > 0 || stod_backward_seconds > 0) {
            cout << "║    STOD:   Fwd " << fixed << setprecision(2) << setw(7) << stod_forward_seconds 
                 << "s | Bwd " << setw(7) << stod_backward_seconds << "s | Total " 
                 << setw(7) << total_stod << "s         ║" << endl;
        }
        if (finstod_forward_seconds > 0 || finstod_backward_seconds > 0) {
            cout << "║    FINSTOD:  Fwd " << fixed << setprecision(2) << setw(7) << finstod_forward_seconds 
                 << "s | Bwd " << setw(7) << finstod_backward_seconds << "s | Total " 
                 << setw(7) << total_finstod << "s         ║" << endl;
        }
        cout << "║    ─────────────────────────────────────────────────────────────────────     ║" << endl;
        cout << "║    STOD Total (Traj + STOD + FINSTOD):                 " 
             << setw(7) << total_stod << "s         ║" << endl;
        cout << "╠══════════════════════════════════════════════════════════════════════════════╣" << endl;
        cout << "║  COMPARISON:                                                                 ║" << endl;
        if (total_standard > 0 && total_stod > 0) {
            double ratio = total_stod / total_standard;
            cout << "║    STOD / Standard = " << fixed << setprecision(2) << ratio << "x";
            if (ratio > 1) {
                cout << "  (STOD is " << setprecision(1) << (ratio - 1) * 100 << "% slower)";
            } else if (ratio < 1) {
                cout << "  (STOD is " << setprecision(1) << (1 - ratio) * 100 << "% faster)";
            } else {
                cout << "  (about equal)";
            }
            cout << string(22, ' ') << "║" << endl;
        }
        if (total_stod > 0 && total_finstod > 0) {
            double ratio_if = total_finstod / total_stod;
            cout << "║    FINSTOD / STOD = " << fixed << setprecision(2) << ratio_if << "x";
            cout << string(44, ' ') << "║" << endl;
        }
        cout << "║    Total computation time: " << setw(7) << total_all << "s";
        cout << string(38, ' ') << "║" << endl;
        cout << "╚══════════════════════════════════════════════════════════════════════════════╝" << endl;
        
        // Parseable format for aggregation
        cout << "[TIMING_DATA]"
             << " FTLE_FWD=" << ftle_forward_seconds 
             << " FTLE_BWD=" << ftle_backward_seconds
             << " FLI_FWD=" << fli_forward_seconds 
             << " FLI_BWD=" << fli_backward_seconds
             << " LD_FWD=" << ld_forward_seconds 
             << " LD_BWD=" << ld_backward_seconds
             << " STOD_TRAJ_FWD=" << stod_traj_forward_seconds 
             << " STOD_TRAJ_BWD=" << stod_traj_backward_seconds
             << " STOD_FWD=" << stod_forward_seconds 
             << " STOD_BWD=" << stod_backward_seconds
             << " FINSTOD_FWD=" << finstod_forward_seconds 
             << " FINSTOD_BWD=" << finstod_backward_seconds
             << " TOTAL_STANDARD=" << total_standard
             << " TOTAL_STOD_TRAJ=" << total_stod_traj
             << " TOTAL_STOD=" << total_stod
             << " TOTAL_FINSTOD=" << total_finstod
             << " TOTAL_STOD=" << total_stod
             << " TOTAL=" << total_all << endl;
    }
};

// =========================================================
// 8. MAIN
// =========================================================
int main(int argc, char* argv[]) {
    // HDF5 error suppression
    H5::Exception::dontPrint();
    
    // Initialize timing stats
    TimingStats timing;

    if (argc < 12) return 1;

    int idx = 1;
    string sys_name = argv[idx++];
    string in_file = argv[idx++];
    string out_file = argv[idx++];
    int nx = stoi(argv[idx++]);
    int ny = stoi(argv[idx++]);
    double x_min = stod(argv[idx++]);
    double x_max = stod(argv[idx++]);
    double y_min = stod(argv[idx++]);
    double y_max = stod(argv[idx++]);
    double t_snap = stod(argv[idx++]);
    double t_total = stod(argv[idx++]);

    // Parse start_row and end_row FIRST (they come right after t_total at positions 12 and 13)
    int start_row = 0;
    int end_row = ny; // default
    if (argc > idx) {
        // Try to parse as integers - if they're not integers, they're probably flags
        // Check if the next argument looks like an integer (doesn't start with '-' and is numeric)
        std::string next_arg = argv[idx];
        if (!next_arg.empty() && next_arg[0] != '-' && 
            (next_arg[0] >= '0' && next_arg[0] <= '9')) {
            try {
                start_row = std::stoi(next_arg);
                if (argc > idx + 1) {
                    std::string next_arg2 = argv[idx + 1];
                    if (!next_arg2.empty() && next_arg2[0] != '-' && 
                        (next_arg2[0] >= '0' && next_arg2[0] <= '9')) {
                        try {
                            end_row = std::stoi(next_arg2);
                            idx += 2; // Skip start_row and end_row
                        } catch (const std::exception& e) {
                            // Second arg is not an integer, so start_row was probably not actually start_row
                            // Reset and don't skip
                            std::cerr << "[WARN] Failed to parse end_row: " << e.what() << ", using defaults" << std::endl;
                            start_row = 0;
                            end_row = ny;
                        }
                    }
                }
            } catch (const std::exception& e) {
                // First arg is not an integer, so start_row/end_row not provided
                // Keep defaults
                std::cerr << "[WARN] Failed to parse start_row: " << e.what() << ", using defaults" << std::endl;
            }
        }
    }

    // Parse extra params (including LB and OLB flags) - skip start_row/end_row if we found them
    std::map<std::string, double> phys_params;
    bool compute_lb = false;
    int lb_slices = 300;
    bool compute_olb_stod = false;
    bool compute_olb_finstod = false;
    int olb_slices = 300;
    
    // Analysis control flags (default to false - only compute if explicitly enabled)
    bool compute_ftle_forward = false;
    bool compute_ftle_backward = false;
    bool compute_stod_forward = false;
    bool compute_stod_backward = false;
    bool compute_finstod_forward = false;
    bool compute_finstod_backward = false;
    
    // FLI and LD flags
    bool compute_fli_forward = false;
    bool compute_fli_backward = false;
    bool compute_ld_forward = false;
    bool compute_ld_backward = false;
    double ld_p_norm = 0.5;           // Default p-norm for LD
    double fli_alpha_rescale = 0.0;   // Rescaling factor to prevent FLI overflow (0 = no rescaling)
    
    // Orthogonal field mode
    bool use_orthogonal = false;      // If true, use orthogonal velocity field (-v, u)
    
    // FTLE method selection (variational is default - avoids finite difference artifacts)
    bool use_variational_ftle = true; // If true, use variational equations; if false, use finite differences

    for (int i = idx; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--compute-lb") {
            compute_lb = true;
        } else if (arg == "--compute-ftle-forward") {
            compute_ftle_forward = true;
        } else if (arg == "--compute-ftle-backward") {
            compute_ftle_backward = true;
        } else if (arg == "--compute-stod-forward") {
            compute_stod_forward = true;
        } else if (arg == "--compute-stod-backward") {
            compute_stod_backward = true;
        } else if (arg == "--compute-finstod-forward") {
            compute_finstod_forward = true;
        } else if (arg == "--compute-finstod-backward") {
            compute_finstod_backward = true;
        } else if (arg.find("lb_slices=") != std::string::npos) {
            try {
                size_t pos = arg.find("lb_slices=");
                lb_slices = std::stoi(arg.substr(pos + 10));
            } catch (...) {
                std::cerr << "[WARN] Failed to parse lb_slices, using default 300" << std::endl;
            }
        } else if (arg == "--compute-olb-stod") {
            compute_olb_stod = true;
        } else if (arg == "--compute-olb-finstod") {
            compute_olb_finstod = true;
        } else if (arg.find("olb_slices=") != std::string::npos) {
            try {
                size_t pos = arg.find("olb_slices=");
                olb_slices = std::stoi(arg.substr(pos + 11));
            } catch (...) {
                std::cerr << "[WARN] Failed to parse olb_slices, using default 300" << std::endl;
            }
        } else if (arg == "--compute-fli-forward") {
            compute_fli_forward = true;
        } else if (arg == "--compute-fli-backward") {
            compute_fli_backward = true;
        } else if (arg == "--compute-ld-forward") {
            compute_ld_forward = true;
        } else if (arg == "--compute-ld-backward") {
            compute_ld_backward = true;
        } else if (arg == "--compute-ld") {
            // Backward compatibility: --compute-ld enables both forward and backward
            compute_ld_forward = true;
            compute_ld_backward = true;
        } else if (arg.find("ld_p_norm=") != std::string::npos) {
            try {
                size_t pos = arg.find("ld_p_norm=");
                ld_p_norm = std::stod(arg.substr(pos + 10));
            } catch (...) {
                std::cerr << "[WARN] Failed to parse ld_p_norm, using default 0.5" << std::endl;
            }
        } else if (arg.find("fli_alpha=") != std::string::npos) {
            try {
                size_t pos = arg.find("fli_alpha=");
                fli_alpha_rescale = std::stod(arg.substr(pos + 10));
            } catch (...) {
                std::cerr << "[WARN] Failed to parse fli_alpha, using default 0.0" << std::endl;
            }
        } else if (arg == "--use-orthogonal") {
            use_orthogonal = true;
        } else if (arg.find("use_orthogonal=") != std::string::npos) {
            try {
                size_t pos = arg.find("use_orthogonal=");
                use_orthogonal = (std::stoi(arg.substr(pos + 15)) != 0);
            } catch (...) {
                std::cerr << "[WARN] Failed to parse use_orthogonal, using default false" << std::endl;
            }
        } else if (arg == "--ftle-finite-diff") {
            // Use finite difference FTLE instead of variational (legacy method)
            use_variational_ftle = false;
        } else if (arg == "--ftle-variational") {
            // Use variational FTLE (default, but explicit flag for clarity)
            use_variational_ftle = true;
        } else {
            size_t eq_pos = arg.find('=');
            if (eq_pos != std::string::npos) {
                try {
                    phys_params[arg.substr(0, eq_pos)] = std::stod(arg.substr(eq_pos + 1));
                } catch(...) {}
            }
        }
    }

    DynamicalSystem* sys = create_system(sys_name, phys_params);
    if (!sys) return 1;

    // Debug output (can be removed in production)
    cout << "[DEBUG] Parsed arguments: start_row=" << start_row << ", end_row=" << end_row 
         << ", compute_ftle_forward=" << compute_ftle_forward << ", compute_ftle_backward=" << compute_ftle_backward
         << ", compute_stod_forward=" << compute_stod_forward << ", compute_stod_backward=" << compute_stod_backward
         << ", compute_finstod_forward=" << compute_finstod_forward << ", compute_finstod_backward=" << compute_finstod_backward
         << ", compute_lb=" << compute_lb << ", compute_olb_stod=" << compute_olb_stod 
         << ", compute_olb_finstod=" << compute_olb_finstod 
         << ", lb_slices=" << lb_slices << ", olb_slices=" << olb_slices 
         << ", compute_fli_forward=" << compute_fli_forward << ", compute_fli_backward=" << compute_fli_backward
         << ", compute_ld_forward=" << compute_ld_forward << ", compute_ld_backward=" << compute_ld_backward
         << ", ld_p_norm=" << ld_p_norm << ", fli_alpha=" << fli_alpha_rescale 
         << ", use_orthogonal=" << use_orthogonal 
         << ", use_variational_ftle=" << use_variational_ftle << endl;

    // Load Data (only to get dimensions - we do fresh integration, not stored trajectories)
    H5File fin;
    DataSet ds_fwd;
    try {
        fin = H5File(in_file, H5F_ACC_RDONLY);
        ds_fwd = fin.openDataSet("forward");
    } catch (const H5::FileIException& e) {
        cerr << "!!! HDF5 File Error: Cannot open input file '" << in_file << "'" << endl;
        cerr << "   Error details: " << e.getDetailMsg() << endl;
        cerr << "   This usually means:" << endl;
        cerr << "   1. The trajectory file doesn't exist or wasn't generated properly" << endl;
        cerr << "   2. The file is corrupted or incomplete" << endl;
        cerr << "   3. There's a permissions issue" << endl;
        cerr << "   Check Stage 1 (trajectory generation) logs for this worker." << endl;
        delete sys;
        return 1;
    } catch (const H5::DataSetIException& e) {
        cerr << "!!! HDF5 Dataset Error: Cannot open 'forward' dataset in file '" << in_file << "'" << endl;
        cerr << "   Error details: " << e.getDetailMsg() << endl;
        cerr << "   The file exists but doesn't contain the required 'forward' dataset." << endl;
        cerr << "   This usually means the trajectory file is incomplete or corrupted." << endl;
        try { fin.close(); } catch (...) {}
        delete sys;
        return 1;
    } catch (const H5::Exception& e) {
        cerr << "!!! HDF5 Error: " << e.getDetailMsg() << endl;
        delete sys;
        return 1;
    }
    
    hsize_t dims[4];
    ds_fwd.getSpace().getSimpleExtentDims(dims, NULL);
    int rows_from_file = dims[0], cols = dims[1], total_steps = dims[2];
    
    // For global metrics (using dummy file), calculate actual rows from start_row/end_row
    // For local metrics, use rows from the actual trajectory file
    int rows;
    if (rows_from_file == 1 && end_row > start_row) {
        // This is likely a dummy file for global metrics - use actual row range
        rows = end_row - start_row;
        cout << "[INFO] Detected dummy file (rows=1), using row range: " << start_row << "-" << end_row 
             << " (actual rows=" << rows << ")" << endl;
    } else {
        // Use rows from the actual trajectory file
        rows = rows_from_file;
    }

    // Get dt from config parameters (preferred) or calculate from stored trajectory data
    // time_step_dt comes from time_params, so check phys_params (which includes all numeric params)
    double dt = 0.01; // default
    if (phys_params.find("time_step_dt") != phys_params.end()) {
        dt = phys_params.at("time_step_dt");
    } else if (total_steps > 1) {
        // Calculate from stored trajectory data as fallback
        dt = t_total / (total_steps - 1);
    }

    // Output Buffers
    vector<double> fwd_ftle(rows*cols), bwd_ftle(rows*cols);
    vector<double> fwd_stod_score(rows*cols), fwd_finstod_score(rows*cols);
    vector<int>    fwd_stod_type(rows*cols), fwd_finstod_type(rows*cols);
    vector<double> bwd_stod_score(rows*cols), bwd_finstod_score(rows*cols);
    vector<int>    bwd_stod_type(rows*cols), bwd_finstod_type(rows*cols);

    // --- 1. FTLE ---
    // Use variational method (default) or finite-difference method
    // Variational method avoids numerical artifacts near invariant manifolds and fixed points
    if (use_variational_ftle) {
        // Forward
        if (compute_ftle_forward && (t_total - t_snap) > 1e-5) {
            auto t_start_ftle_fwd = chrono::high_resolution_clock::now();
            compute_variational_ftle_slice(sys, rows, cols, start_row, nx, ny, x_min, x_max, y_min, y_max, 
                                           t_snap, t_total, fli_alpha_rescale, fwd_ftle, "Forward", use_orthogonal);
            auto t_end_ftle_fwd = chrono::high_resolution_clock::now();
            timing.ftle_forward_seconds += chrono::duration<double>(t_end_ftle_fwd - t_start_ftle_fwd).count();
        }
        // Backward
        if (compute_ftle_backward && t_snap > 1e-5) {
            auto t_start_ftle_bwd = chrono::high_resolution_clock::now();
            compute_variational_ftle_slice(sys, rows, cols, start_row, nx, ny, x_min, x_max, y_min, y_max, 
                                           t_snap, 0.0, fli_alpha_rescale, bwd_ftle, "Backward", use_orthogonal);
            auto t_end_ftle_bwd = chrono::high_resolution_clock::now();
            timing.ftle_backward_seconds += chrono::duration<double>(t_end_ftle_bwd - t_start_ftle_bwd).count();
        }
    } else {
        // Legacy finite-difference method (may show artifacts near manifolds)
        // Forward
        if (compute_ftle_forward && (t_total - t_snap) > 1e-5) {
            auto t_start_ftle_fwd = chrono::high_resolution_clock::now();
            compute_fresh_ftle_slice(sys, rows, cols, start_row, nx, ny, x_min, x_max, y_min, y_max, t_snap, t_total, fwd_ftle, "Forward", use_orthogonal);
            auto t_end_ftle_fwd = chrono::high_resolution_clock::now();
            timing.ftle_forward_seconds += chrono::duration<double>(t_end_ftle_fwd - t_start_ftle_fwd).count();
        }
        // Backward
        if (compute_ftle_backward && t_snap > 1e-5) {
            auto t_start_ftle_bwd = chrono::high_resolution_clock::now();
            compute_fresh_ftle_slice(sys, rows, cols, start_row, nx, ny, x_min, x_max, y_min, y_max, t_snap, 0.0, bwd_ftle, "Backward", use_orthogonal);
            auto t_end_ftle_bwd = chrono::high_resolution_clock::now();
            timing.ftle_backward_seconds += chrono::duration<double>(t_end_ftle_bwd - t_start_ftle_bwd).count();
        }
    }

    // --- 2. STOD/FINSTOD (Fresh Integration from t_snap, matching Python approach) ---
    // Forward: integrate from t_snap to t_total
    if ((compute_stod_forward || compute_finstod_forward) && (t_total - t_snap) > 1e-5) {
        SmodTimingBreakdown stod_timing_fwd = compute_fresh_stod_finstod_slice(
            sys, rows, cols, start_row, nx, ny, x_min, x_max, y_min, y_max,
            t_snap, t_total, dt,
            fwd_stod_score, fwd_stod_type, fwd_finstod_score, fwd_finstod_type, "Forward", use_orthogonal);
        timing.stod_traj_forward_seconds += stod_timing_fwd.trajectory_seconds;
        timing.stod_forward_seconds += stod_timing_fwd.stod_seconds;
        timing.finstod_forward_seconds += stod_timing_fwd.finstod_seconds;
    }
    
    // Backward: integrate from t_snap to 0.0
    if ((compute_stod_backward || compute_finstod_backward) && t_snap > 1e-5) {
        SmodTimingBreakdown stod_timing_bwd = compute_fresh_stod_finstod_slice(
            sys, rows, cols, start_row, nx, ny, x_min, x_max, y_min, y_max,
            t_snap, 0.0, dt,
            bwd_stod_score, bwd_stod_type, bwd_finstod_score, bwd_finstod_type, "Backward", use_orthogonal);
        timing.stod_traj_backward_seconds += stod_timing_bwd.trajectory_seconds;
        timing.stod_backward_seconds += stod_timing_bwd.stod_seconds;
        timing.finstod_backward_seconds += stod_timing_bwd.finstod_seconds;
    }

    // --- 3. Exact LB (Optional) ---
    // Only calculate if flag is set. Usually done on one snapshot or all, 
    // but LB is time-independent (integral). We calculate it if requested.
    vector<double> lb_data_fxb, lb_data_fpb, lb_data_fmb;
    if (compute_lb) {
        lb_data_fxb.resize(rows*cols);
        lb_data_fpb.resize(rows*cols);
        lb_data_fmb.resize(rows*cols);
        compute_exact_lb_slice(sys, rows, cols, start_row, nx, ny, x_min, x_max, y_min, y_max, 
                               t_total, lb_slices, lb_data_fxb, lb_data_fpb, lb_data_fmb, use_orthogonal);
    }

    // --- 4. Exact OLB (Optional) ---
    // OLB using STOD or FINSTOD, computed using CACHED trajectories for major speedup!
    // This reduces ~3000 integrations per pixel to just 5 (center + 4 neighbors)
    vector<double> olb_stod_data_fxb, olb_stod_data_fpb, olb_stod_data_fmb;
    vector<double> olb_finstod_data_fxb, olb_finstod_data_fpb, olb_finstod_data_fmb;
    if (compute_olb_stod) {
        olb_stod_data_fxb.resize(rows*cols);
        olb_stod_data_fpb.resize(rows*cols);
        olb_stod_data_fmb.resize(rows*cols);
        // Use cached v2 for autonomous systems (correct + fast)
        compute_exact_olb_slice_cached_v2(sys, rows, cols, start_row, nx, ny, x_min, x_max, y_min, y_max, 
                                          t_total, dt, olb_slices, false, 
                                          olb_stod_data_fxb, olb_stod_data_fpb, olb_stod_data_fmb, use_orthogonal);
    }
    if (compute_olb_finstod) {
        olb_finstod_data_fxb.resize(rows*cols);
        olb_finstod_data_fpb.resize(rows*cols);
        olb_finstod_data_fmb.resize(rows*cols);
        // Use cached v2 for autonomous systems (correct + fast)
        compute_exact_olb_slice_cached_v2(sys, rows, cols, start_row, nx, ny, x_min, x_max, y_min, y_max, 
                                          t_total, dt, olb_slices, true, 
                                          olb_finstod_data_fxb, olb_finstod_data_fpb, olb_finstod_data_fmb, use_orthogonal);
    }

    // --- 5. FLI (Fast Lyapunov Indicator) using Variational Equations ---
    vector<double> fwd_fli_var(rows*cols), bwd_fli_var(rows*cols);
    // Forward FLI: from t_snap to t_total
    if (compute_fli_forward && (t_total - t_snap) > 1e-5) {
        auto t_start_fli_fwd = chrono::high_resolution_clock::now();
        compute_fli_slice(sys, rows, cols, start_row, nx, ny, x_min, x_max, y_min, y_max, 
                          t_snap, t_total, fli_alpha_rescale, fwd_fli_var, "Forward", use_orthogonal);
        auto t_end_fli_fwd = chrono::high_resolution_clock::now();
        timing.fli_forward_seconds += chrono::duration<double>(t_end_fli_fwd - t_start_fli_fwd).count();
    }
    // Backward FLI: from t_snap to 0
    if (compute_fli_backward && t_snap > 1e-5) {
        auto t_start_fli_bwd = chrono::high_resolution_clock::now();
        compute_fli_slice(sys, rows, cols, start_row, nx, ny, x_min, x_max, y_min, y_max, 
                          t_snap, 0.0, fli_alpha_rescale, bwd_fli_var, "Backward", use_orthogonal);
        auto t_end_fli_bwd = chrono::high_resolution_clock::now();
        timing.fli_backward_seconds += chrono::duration<double>(t_end_fli_bwd - t_start_fli_bwd).count();
    }

    // --- 6. LD (Lagrangian Descriptor) ---
    vector<double> fwd_ld(rows*cols), bwd_ld(rows*cols);
    if (compute_ld_forward || compute_ld_backward) {
        auto t_start_ld = chrono::high_resolution_clock::now();
        compute_ld_slice(sys, rows, cols, start_row, nx, ny, x_min, x_max, y_min, y_max, 
                         t_snap, t_total, ld_p_norm, compute_ld_forward, compute_ld_backward,
                         fwd_ld, bwd_ld, "Full", use_orthogonal);
        auto t_end_ld = chrono::high_resolution_clock::now();
        double ld_total_time = chrono::duration<double>(t_end_ld - t_start_ld).count();
        // Split time between forward and backward based on what was computed
        if (compute_ld_forward && compute_ld_backward) {
            timing.ld_forward_seconds += ld_total_time / 2.0;
            timing.ld_backward_seconds += ld_total_time / 2.0;
        } else if (compute_ld_forward) {
            timing.ld_forward_seconds += ld_total_time;
        } else {
            timing.ld_backward_seconds += ld_total_time;
        }
    }

    // --- Print timing summary ---
    timing.print_summary();

    // --- Write Output ---
    try {
        H5File fout(out_file, H5F_ACC_TRUNC);
        hsize_t dim_out[2] = {(hsize_t)rows, (hsize_t)cols};
        DataSpace space_out(2, dim_out);
        
        fout.createDataSet("fwd_ftle", PredType::NATIVE_DOUBLE, space_out).write(fwd_ftle.data(), PredType::NATIVE_DOUBLE);
        fout.createDataSet("bwd_ftle", PredType::NATIVE_DOUBLE, space_out).write(bwd_ftle.data(), PredType::NATIVE_DOUBLE);
        fout.createDataSet("fwd_stod_score", PredType::NATIVE_DOUBLE, space_out).write(fwd_stod_score.data(), PredType::NATIVE_DOUBLE);
        fout.createDataSet("fwd_stod_type", PredType::NATIVE_INT, space_out).write(fwd_stod_type.data(), PredType::NATIVE_INT);
        fout.createDataSet("fwd_finstod_score", PredType::NATIVE_DOUBLE, space_out).write(fwd_finstod_score.data(), PredType::NATIVE_DOUBLE);
        fout.createDataSet("fwd_finstod_type", PredType::NATIVE_INT, space_out).write(fwd_finstod_type.data(), PredType::NATIVE_INT);
        fout.createDataSet("bwd_stod_score", PredType::NATIVE_DOUBLE, space_out).write(bwd_stod_score.data(), PredType::NATIVE_DOUBLE);
        fout.createDataSet("bwd_stod_type", PredType::NATIVE_INT, space_out).write(bwd_stod_type.data(), PredType::NATIVE_INT);
        fout.createDataSet("bwd_finstod_score", PredType::NATIVE_DOUBLE, space_out).write(bwd_finstod_score.data(), PredType::NATIVE_DOUBLE);
        fout.createDataSet("bwd_finstod_type", PredType::NATIVE_INT, space_out).write(bwd_finstod_type.data(), PredType::NATIVE_INT);

        if (compute_lb) {
            fout.createDataSet("exact_lb_fxb", PredType::NATIVE_DOUBLE, space_out).write(lb_data_fxb.data(), PredType::NATIVE_DOUBLE);
            fout.createDataSet("exact_lb_fpb", PredType::NATIVE_DOUBLE, space_out).write(lb_data_fpb.data(), PredType::NATIVE_DOUBLE);
            fout.createDataSet("exact_lb_fmb", PredType::NATIVE_DOUBLE, space_out).write(lb_data_fmb.data(), PredType::NATIVE_DOUBLE);
            // Keep backward compatibility: exact_lb = exact_lb_fxb (classical LB)
            fout.createDataSet("exact_lb", PredType::NATIVE_DOUBLE, space_out).write(lb_data_fxb.data(), PredType::NATIVE_DOUBLE);
        }
        
        if (compute_olb_stod) {
            fout.createDataSet("exact_olb_stod_fxb", PredType::NATIVE_DOUBLE, space_out).write(olb_stod_data_fxb.data(), PredType::NATIVE_DOUBLE);
            fout.createDataSet("exact_olb_stod_fpb", PredType::NATIVE_DOUBLE, space_out).write(olb_stod_data_fpb.data(), PredType::NATIVE_DOUBLE);
            fout.createDataSet("exact_olb_stod_fmb", PredType::NATIVE_DOUBLE, space_out).write(olb_stod_data_fmb.data(), PredType::NATIVE_DOUBLE);
            // Keep backward compatibility: exact_olb_stod = exact_olb_stod_fxb (classical OLB)
            fout.createDataSet("exact_olb_stod", PredType::NATIVE_DOUBLE, space_out).write(olb_stod_data_fxb.data(), PredType::NATIVE_DOUBLE);
        }
        
        if (compute_olb_finstod) {
            fout.createDataSet("exact_olb_finstod_fxb", PredType::NATIVE_DOUBLE, space_out).write(olb_finstod_data_fxb.data(), PredType::NATIVE_DOUBLE);
            fout.createDataSet("exact_olb_finstod_fpb", PredType::NATIVE_DOUBLE, space_out).write(olb_finstod_data_fpb.data(), PredType::NATIVE_DOUBLE);
            fout.createDataSet("exact_olb_finstod_fmb", PredType::NATIVE_DOUBLE, space_out).write(olb_finstod_data_fmb.data(), PredType::NATIVE_DOUBLE);
            // Keep backward compatibility: exact_olb_finstod = exact_olb_finstod_fxb (classical OLB)
            fout.createDataSet("exact_olb_finstod", PredType::NATIVE_DOUBLE, space_out).write(olb_finstod_data_fxb.data(), PredType::NATIVE_DOUBLE);
        }

        // FLI (Fast Lyapunov Indicator) using variational equations
        if (compute_fli_forward) {
            fout.createDataSet("fwd_fli", PredType::NATIVE_DOUBLE, space_out).write(fwd_fli_var.data(), PredType::NATIVE_DOUBLE);
        }
        if (compute_fli_backward) {
            fout.createDataSet("bwd_fli", PredType::NATIVE_DOUBLE, space_out).write(bwd_fli_var.data(), PredType::NATIVE_DOUBLE);
        }

        // LD (Lagrangian Descriptor) - Forward and Backward separately
        if (compute_ld_forward) {
            fout.createDataSet("fwd_ld", PredType::NATIVE_DOUBLE, space_out).write(fwd_ld.data(), PredType::NATIVE_DOUBLE);
        }
        if (compute_ld_backward) {
            fout.createDataSet("bwd_ld", PredType::NATIVE_DOUBLE, space_out).write(bwd_ld.data(), PredType::NATIVE_DOUBLE);
        }

        fout.close();
    } catch (const H5::FileIException& e) {
        cerr << "!!! HDF5 File Error: Cannot create output file '" << out_file << "'" << endl;
        cerr << "   Error details: " << e.getDetailMsg() << endl;
        cerr << "   Check directory permissions and disk space." << endl;
        fin.close();
        delete sys;
        return 1;
    } catch (const H5::Exception& e) {
        cerr << "!!! HDF5 Error while writing output: " << e.getDetailMsg() << endl;
        fin.close();
        delete sys;
        return 1;
    }
    
    fin.close();
    delete sys;
    return 0;
}
