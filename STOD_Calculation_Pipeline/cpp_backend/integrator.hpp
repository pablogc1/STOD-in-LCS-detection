#ifndef INTEGRATOR_HPP
#define INTEGRATOR_HPP

#include "systems.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

// RKF45 Coefficients (Cash-Karp parameters are efficient)
static const double c2 = 1.0/5.0;
static const double c3 = 3.0/10.0;
static const double c4 = 3.0/5.0;
static const double c5 = 1.0;
static const double c6 = 7.0/8.0;

static const double a21 = 1.0/5.0;
static const double a31 = 3.0/40.0;     static const double a32 = 9.0/40.0;
static const double a41 = 3.0/10.0;     static const double a42 = -9.0/10.0;    static const double a43 = 6.0/5.0;
static const double a51 = -11.0/54.0;   static const double a52 = 5.0/2.0;      static const double a53 = -70.0/27.0;   static const double a54 = 35.0/27.0;
static const double a61 = 1631.0/55296.0;static const double a62 = 175.0/512.0; static const double a63 = 575.0/13824.0;  static const double a64 = 44275.0/110592.0; static const double a65 = 253.0/4096.0;

// 5th Order solution weights
static const double b1 = 37.0/378.0;    static const double b3 = 250.0/621.0;   static const double b4 = 125.0/594.0;   static const double b6 = 512.0/1771.0;
// 4th Order solution weights (for error estimation)
static const double bs1 = 2825.0/27648.0;static const double bs3 = 18575.0/48384.0;static const double bs4 = 13525.0/55296.0;static const double bs5 = 277.0/14336.0;static const double bs6 = 1.0/4.0;

// Performs integration with ADAPTIVE step size (RKF45).
// It attempts to hit specific output times (t_start + n*dt_out) for uniform sampling,
// but takes adaptive substeps internally.
// use_orthogonal: if true, integrates using the orthogonal velocity field (-v, u)
void integrate_path_adaptive(
    const DynamicalSystem* sys,
    const std::vector<double>& start_state,
    double t_start, double t_end, double dt_out, // dt_out is the DESIRED output spacing
    double direction, 
    std::vector<double>& out_path,
    bool use_orthogonal = false
) {
    int dim = sys->get_dim();
    
    // Configurable Tolerances (matched to Fortran reference: relerr=1e-8, abserr=1e-8)
    const double atol = 1e-8;
    const double rtol = 1e-8;
    const double SAFETY = 0.9;
    const double MIN_SCALE = 0.1;
    const double MAX_SCALE = 5.0;
    
    std::vector<double> y = start_state;
    std::vector<double> y_temp(dim), y_err(dim);
    std::vector<double> k1(dim), k2(dim), k3(dim), k4(dim), k5(dim), k6(dim);

    // Initial step guess (small fraction of output step)
    double h = (direction > 0 ? 1.0 : -1.0) * std::abs(dt_out) * 0.1;
    double t = t_start;
    double t_final = t_end;
    
    // Ensure we capture start point
    out_path.insert(out_path.end(), y.begin(), y.end());
    
    double t_next_output = t_start + (direction * std::abs(dt_out));
    
    // Loop until we pass t_final
    bool done = false;
    while (!done) {
        
        // Check if we are close to end
        double dist_to_end = std::abs(t_final - t);
        if (dist_to_end < 1e-12) break;
        
        // 1. Determine step size for this attempt
        // Clamp h so we don't overshoot the next OUTPUT point too wildly
        // (Optional, but helps keep output grids uniform-ish for SOD)
        double dist_to_next_out = std::abs(t_next_output - t);
        
        // If the adaptive step is larger than distance to output, shrink it to hit output exactly
        if (std::abs(h) > dist_to_next_out) {
            h = (direction > 0 ? 1.0 : -1.0) * dist_to_next_out;
        }

        // 2. Take RKF45 Step
        // Use orthogonal velocity if requested
        auto get_vel = [&](double t_val, const std::vector<double>& state, std::vector<double>& out) {
            if (use_orthogonal) {
                sys->get_velocity_orthogonal(t_val, state, out);
            } else {
                sys->get_velocity(t_val, state, out);
            }
        };
        
        get_vel(t, y, k1);
        
        for(int i=0; i<dim; ++i) y_temp[i] = y[i] + h * (a21*k1[i]);
        get_vel(t + c2*h, y_temp, k2);
        
        for(int i=0; i<dim; ++i) y_temp[i] = y[i] + h * (a31*k1[i] + a32*k2[i]);
        get_vel(t + c3*h, y_temp, k3);
        
        for(int i=0; i<dim; ++i) y_temp[i] = y[i] + h * (a41*k1[i] + a42*k2[i] + a43*k3[i]);
        get_vel(t + c4*h, y_temp, k4);
        
        for(int i=0; i<dim; ++i) y_temp[i] = y[i] + h * (a51*k1[i] + a52*k2[i] + a53*k3[i] + a54*k4[i]);
        get_vel(t + c5*h, y_temp, k5);
        
        for(int i=0; i<dim; ++i) y_temp[i] = y[i] + h * (a61*k1[i] + a62*k2[i] + a63*k3[i] + a64*k4[i] + a65*k5[i]);
        get_vel(t + c6*h, y_temp, k6);

        // 3. Error Estimate
        double max_err = 0.0;
        for(int i=0; i<dim; ++i) {
            // Difference between 4th and 5th order
            double diff = h * ((b1-bs1)*k1[i] + (b3-bs3)*k3[i] + (b4-bs4)*k4[i] + (b6-bs6)*k6[i] - bs5*k5[i]); // Note: b5 is 0
            
            // Scaling tolerance: atol + rtol * |y|
            double scale = atol + rtol * std::max(std::abs(y[i]), std::abs(y_temp[i])); // approximate y_next via y_temp
            double ratio = std::abs(diff) / scale;
            if (ratio > max_err) max_err = ratio;
        }

        // 4. Accept or Reject
        if (max_err <= 1.0) {
            // ACCEPT
            t += h;
            // Calculate y_new (5th order)
            for(int i=0; i<dim; ++i) {
                y[i] += h * (b1*k1[i] + b3*k3[i] + b4*k4[i] + b6*k6[i]); // b2=0, b5=0
            }
            
            // If we hit the output time (or passed it very slightly due to float precision)
            if (dist_to_next_out < 1e-12 || (direction > 0 && t >= t_next_output) || (direction < 0 && t <= t_next_output)) {
                out_path.insert(out_path.end(), y.begin(), y.end());
                t_next_output += (direction * std::abs(dt_out));
            }
            
            // Check if finished
             if ((direction > 0 && t >= t_final) || (direction < 0 && t <= t_final)) {
                done = true;
            }

        } 
        
        // 5. Adjust Step Size (Adaptive) for next try (or next step)
        // h_opt = h * SAFETY * (1/err)^0.2
        double scale_factor = 0.0;
        if (max_err == 0.0) scale_factor = MAX_SCALE;
        else scale_factor = SAFETY * std::pow(max_err, -0.2); // Power 1/5 for 4th order method
        
        scale_factor = std::max(MIN_SCALE, std::min(MAX_SCALE, scale_factor));
        
        // If step was rejected, we reduce h and try same step again. 
        // If accepted, we update h for the NEXT step.
        h *= scale_factor;
        
        // Prevent h from becoming too small (stiff crash check)
        if (std::abs(h) < 1e-15) {
             // Force move or abort? For robust generic pipeline, we abort this particle.
             // But to keep output consistency, we might just stop this trajectory here.
             break;
        }
    }
}

// =========================================================
// VARIATIONAL INTEGRATOR FOR FLI AND LD
// =========================================================
// Integrates the state AND the tangent matrix (Φ) simultaneously.
// Also accumulates the LD integral (sum of velocity norms).
//
// The variational equations are:
//   d(state)/dt = v(t, state)
//   d(Phi)/dt = J(t, state) * Phi
//
// where J is the Jacobian matrix and Phi starts as identity.
// 
// For FLI: We track the maximum column norm of Phi (after optional rescaling).
// For LD:  We integrate ||v||_p along the trajectory.

struct VariationalResult {
    double fli_value;           // Final FLI = log(max column norm of Phi)
    double ftle_value;          // Final FTLE = log(sqrt(lambda_max(Phi^T Phi))) / T (NOT normalized by T here)
    double ld_value;            // Lagrangian Descriptor integral
    std::vector<double> final_state;
    std::vector<double> final_phi;  // dim x dim tangent matrix (row-major)
};

// Compute the maximum column norm of a dim x dim matrix (row-major)
double max_column_norm(const std::vector<double>& mat, int dim) {
    double max_norm = 0.0;
    for (int j = 0; j < dim; ++j) {  // For each column
        double col_norm = 0.0;
        for (int i = 0; i < dim; ++i) {  // Sum over rows
            double val = mat[i * dim + j];
            col_norm += val * val;
        }
        col_norm = std::sqrt(col_norm);
        if (col_norm > max_norm) max_norm = col_norm;
    }
    return max_norm;
}

// Compute the maximum eigenvalue of the Cauchy-Green tensor C = Phi^T * Phi
// For a 2x2 Phi, this gives the square of the maximum singular value of Phi.
// Returns lambda_max(C), where FTLE = log(lambda_max) / (2*T)
double cauchy_green_max_eigenvalue(const std::vector<double>& phi, int dim) {
    if (dim == 2) {
        // Phi = [[a, b], [c, d]] (row-major: phi[0]=a, phi[1]=b, phi[2]=c, phi[3]=d)
        double a = phi[0], b = phi[1], c = phi[2], d = phi[3];
        
        // C = Phi^T * Phi = [[a*a + c*c, a*b + c*d], [a*b + c*d, b*b + d*d]]
        double c11 = a*a + c*c;
        double c12 = a*b + c*d;
        double c22 = b*b + d*d;
        
        // Eigenvalues of 2x2 symmetric matrix: lambda = (trace ± sqrt(trace^2 - 4*det)) / 2
        double trace = c11 + c22;
        double det = c11 * c22 - c12 * c12;
        double discriminant = trace * trace - 4.0 * det;
        
        if (discriminant < 0.0) discriminant = 0.0;  // Numerical safety
        
        double lambda_max = (trace + std::sqrt(discriminant)) / 2.0;
        return lambda_max;
    }
    else {
        // For higher dimensions, compute C = Phi^T * Phi and find max eigenvalue
        // using power iteration or direct formula for small dims
        std::vector<double> C(dim * dim, 0.0);
        
        // C[i][j] = sum_k Phi[k][i] * Phi[k][j] (Phi^T * Phi)
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                double sum = 0.0;
                for (int k = 0; k < dim; ++k) {
                    sum += phi[k * dim + i] * phi[k * dim + j];
                }
                C[i * dim + j] = sum;
            }
        }
        
        // Power iteration to find maximum eigenvalue
        std::vector<double> v(dim, 1.0 / std::sqrt((double)dim));  // Initial unit vector
        std::vector<double> Cv(dim);
        double lambda = 0.0;
        
        for (int iter = 0; iter < 100; ++iter) {
            // Cv = C * v
            for (int i = 0; i < dim; ++i) {
                Cv[i] = 0.0;
                for (int j = 0; j < dim; ++j) {
                    Cv[i] += C[i * dim + j] * v[j];
                }
            }
            
            // Compute norm of Cv
            double norm = 0.0;
            for (int i = 0; i < dim; ++i) norm += Cv[i] * Cv[i];
            norm = std::sqrt(norm);
            
            if (norm < 1e-15) break;
            
            // Rayleigh quotient: lambda = v^T * C * v (with normalized v)
            double new_lambda = 0.0;
            for (int i = 0; i < dim; ++i) new_lambda += v[i] * Cv[i];
            
            // Normalize v = Cv / ||Cv||
            for (int i = 0; i < dim; ++i) v[i] = Cv[i] / norm;
            
            // Check convergence
            if (std::abs(new_lambda - lambda) < 1e-12 * std::abs(lambda)) {
                lambda = new_lambda;
                break;
            }
            lambda = new_lambda;
        }
        
        return lambda;
    }
}

// Compute L2 norm of velocity vector
double velocity_norm_l2(const std::vector<double>& v) {
    double sum = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        sum += v[i] * v[i];
    }
    return std::sqrt(sum);
}

// Compute Lp norm of velocity vector (for generalized LD)
double velocity_norm_lp(const std::vector<double>& v, double p) {
    if (p == 2.0) return velocity_norm_l2(v);
    double sum = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        sum += std::pow(std::abs(v[i]), p);
    }
    return std::pow(sum, 1.0 / p);
}

// Integrate with variational equations using RKF45.
// Returns FLI and LD values.
// alpha_rescale: if > 0, rescale Phi by exp(-alpha*dt) at each step to prevent overflow
// use_orthogonal: if true, uses orthogonal velocity field (-v, u)
VariationalResult integrate_variational(
    const DynamicalSystem* sys,
    const std::vector<double>& start_state,
    double t_start, double t_end,
    double alpha_rescale = 0.0,  // Rescaling factor (0 = no rescaling, use for very long times)
    double ld_p_norm = 0.5,      // p-norm for LD (0.5 is common, 1.0 = arc length)
    bool use_orthogonal = false  // Use orthogonal velocity field
) {
    int dim = sys->get_dim();
    int dim2 = dim * dim;
    
    // Configurable Tolerances (matched to Fortran reference: relerr=1e-8, abserr=1e-8)
    const double atol = 1e-8;
    const double rtol = 1e-8;
    const double SAFETY = 0.9;
    const double MIN_SCALE = 0.1;
    const double MAX_SCALE = 5.0;
    
    // Combined state: [state (dim), Phi (dim*dim)]
    int total_dim = dim + dim2;
    std::vector<double> y(total_dim);
    
    // Initialize state
    for (int i = 0; i < dim; ++i) y[i] = start_state[i];
    
    // Initialize Phi as identity matrix
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            y[dim + i * dim + j] = (i == j) ? 1.0 : 0.0;
        }
    }
    
    // Workspace
    std::vector<double> y_temp(total_dim), y_err(total_dim);
    std::vector<double> k1(total_dim), k2(total_dim), k3(total_dim);
    std::vector<double> k4(total_dim), k5(total_dim), k6(total_dim);
    std::vector<double> vel(dim), jac(dim2), temp_state(dim);
    
    double direction = (t_end > t_start) ? 1.0 : -1.0;
    double dt_total = std::abs(t_end - t_start);
    double h = direction * dt_total * 0.001;  // Initial step guess
    double t = t_start;
    
    // LD accumulator
    double ld_integral = 0.0;
    double accumulated_log_scale = 0.0;  // For FLI with rescaling
    
    // Lambda to compute derivatives of combined state
    auto compute_deriv = [&](double t_cur, const std::vector<double>& state_in, std::vector<double>& deriv_out) {
        // Extract current state
        for (int i = 0; i < dim; ++i) temp_state[i] = state_in[i];
        
        // Get velocity and Jacobian (use orthogonal versions if requested)
        if (use_orthogonal) {
            sys->get_velocity_orthogonal(t_cur, temp_state, vel);
            sys->get_jacobian_orthogonal(t_cur, temp_state, jac);
        } else {
            sys->get_velocity(t_cur, temp_state, vel);
            sys->get_jacobian(t_cur, temp_state, jac);
        }
        
        // Derivative of state = velocity
        for (int i = 0; i < dim; ++i) deriv_out[i] = vel[i];
        
        // Derivative of Phi = J * Phi
        // Phi is stored row-major at state_in[dim + ...]
        // dPhi[i][j]/dt = sum_k J[i][k] * Phi[k][j]
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                double sum = 0.0;
                for (int k = 0; k < dim; ++k) {
                    sum += jac[i * dim + k] * state_in[dim + k * dim + j];
                }
                deriv_out[dim + i * dim + j] = sum;
            }
        }
    };
    
    // Main integration loop
    while (true) {
        double dist_to_end = std::abs(t_end - t);
        if (dist_to_end < 1e-12) break;
        
        // Clamp step to not overshoot
        if (std::abs(h) > dist_to_end) {
            h = direction * dist_to_end;
        }
        
        // RKF45 step
        compute_deriv(t, y, k1);
        
        for (int i = 0; i < total_dim; ++i) y_temp[i] = y[i] + h * (a21 * k1[i]);
        compute_deriv(t + c2 * h, y_temp, k2);
        
        for (int i = 0; i < total_dim; ++i) y_temp[i] = y[i] + h * (a31 * k1[i] + a32 * k2[i]);
        compute_deriv(t + c3 * h, y_temp, k3);
        
        for (int i = 0; i < total_dim; ++i) y_temp[i] = y[i] + h * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]);
        compute_deriv(t + c4 * h, y_temp, k4);
        
        for (int i = 0; i < total_dim; ++i) y_temp[i] = y[i] + h * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]);
        compute_deriv(t + c5 * h, y_temp, k5);
        
        for (int i = 0; i < total_dim; ++i) y_temp[i] = y[i] + h * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i]);
        compute_deriv(t + c6 * h, y_temp, k6);
        
        // Error estimate
        double max_err = 0.0;
        for (int i = 0; i < total_dim; ++i) {
            double diff = h * ((b1 - bs1) * k1[i] + (b3 - bs3) * k3[i] + (b4 - bs4) * k4[i] + (b6 - bs6) * k6[i] - bs5 * k5[i]);
            double scale = atol + rtol * std::max(std::abs(y[i]), std::abs(y_temp[i]));
            double ratio = std::abs(diff) / scale;
            if (ratio > max_err) max_err = ratio;
        }
        
        // Accept or reject
        if (max_err <= 1.0) {
            // ACCEPT
            double h_abs = std::abs(h);
            t += h;
            
            // Update state (5th order solution)
            for (int i = 0; i < total_dim; ++i) {
                y[i] += h * (b1 * k1[i] + b3 * k3[i] + b4 * k4[i] + b6 * k6[i]);
            }
            
            // Accumulate LD (velocity norm * dt)
            for (int i = 0; i < dim; ++i) temp_state[i] = y[i];
            if (use_orthogonal) {
                sys->get_velocity_orthogonal(t, temp_state, vel);
            } else {
                sys->get_velocity(t, temp_state, vel);
            }
            ld_integral += velocity_norm_lp(vel, ld_p_norm) * h_abs;
            
            // Optional rescaling of Phi to prevent overflow
            if (alpha_rescale > 0.0) {
                double scale_factor = std::exp(-alpha_rescale * h_abs);
                for (int i = 0; i < dim2; ++i) {
                    y[dim + i] *= scale_factor;
                }
                accumulated_log_scale += alpha_rescale * h_abs;
            }
            
            // Check if finished
            if ((direction > 0 && t >= t_end) || (direction < 0 && t <= t_end)) {
                break;
            }
        }
        
        // Adjust step size
        double scale_factor;
        if (max_err == 0.0) scale_factor = MAX_SCALE;
        else scale_factor = SAFETY * std::pow(max_err, -0.2);
        scale_factor = std::max(MIN_SCALE, std::min(MAX_SCALE, scale_factor));
        h *= scale_factor;
        
        // Prevent h from becoming too small
        if (std::abs(h) < 1e-15) break;
    }
    
    // Extract results
    VariationalResult result;
    result.final_state.resize(dim);
    result.final_phi.resize(dim2);
    
    for (int i = 0; i < dim; ++i) result.final_state[i] = y[i];
    for (int i = 0; i < dim2; ++i) result.final_phi[i] = y[dim + i];
    
    // Compute FLI = log(max column norm of Phi) + accumulated_log_scale
    double max_norm = max_column_norm(result.final_phi, dim);
    if (max_norm > 0.0) {
        result.fli_value = std::log(max_norm) + accumulated_log_scale;
    } else {
        result.fli_value = 0.0;
    }
    
    // Compute FTLE from Cauchy-Green tensor: FTLE = log(lambda_max) / (2*T)
    // Here we compute log(lambda_max) / 2; the caller divides by T
    // Note: accumulated_log_scale affects Phi, so lambda_max is scaled by exp(-2*accumulated_log_scale)
    // Thus: log(true_lambda_max) = log(lambda_max) + 2*accumulated_log_scale
    double lambda_max = cauchy_green_max_eigenvalue(result.final_phi, dim);
    if (lambda_max > 0.0) {
        result.ftle_value = (std::log(lambda_max) + 2.0 * accumulated_log_scale) / 2.0;
    } else {
        result.ftle_value = 0.0;
    }
    
    result.ld_value = ld_integral;
    
    return result;
}

// =========================================================
// SIMPLE LD INTEGRATOR (WITHOUT VARIATIONAL EQUATIONS)
// =========================================================
// This integrates ONLY the state equations while accumulating the LD.
// Much faster than integrate_variational for LD since it doesn't compute
// the tangent matrix (dim ODEs instead of dim + dim^2 ODEs).
//
// For a 4D system like LiCN: 4 ODEs instead of 20 ODEs!
// For a 2D system: 2 ODEs instead of 6 ODEs!

struct LDResult {
    double ld_value;                  // Lagrangian Descriptor integral
    std::vector<double> final_state;  // Final state after integration
};

LDResult integrate_path_with_ld(
    const DynamicalSystem* sys,
    const std::vector<double>& start_state,
    double t_start, double t_end,
    double ld_p_norm = 0.5,      // p-norm for LD (0.5 is common, 1.0 = arc length)
    bool use_orthogonal = false  // Use orthogonal velocity field
) {
    int dim = sys->get_dim();
    
    // Configurable Tolerances (matched to Fortran reference: relerr=1e-8, abserr=1e-8)
    const double atol = 1e-8;
    const double rtol = 1e-8;
    const double SAFETY = 0.9;
    const double MIN_SCALE = 0.1;
    const double MAX_SCALE = 5.0;
    
    std::vector<double> y = start_state;
    std::vector<double> y_temp(dim), y_err(dim);
    std::vector<double> k1(dim), k2(dim), k3(dim), k4(dim), k5(dim), k6(dim);
    std::vector<double> vel(dim);
    
    double direction = (t_end > t_start) ? 1.0 : -1.0;
    double dt_total = std::abs(t_end - t_start);
    double h = direction * dt_total * 0.001;  // Initial step guess
    double t = t_start;
    
    // LD accumulator
    double ld_integral = 0.0;
    
    // Lambda to compute velocity
    auto get_vel = [&](double t_val, const std::vector<double>& state, std::vector<double>& out) {
        if (use_orthogonal) {
            sys->get_velocity_orthogonal(t_val, state, out);
        } else {
            sys->get_velocity(t_val, state, out);
        }
    };
    
    // Main integration loop
    while (true) {
        double dist_to_end = std::abs(t_end - t);
        if (dist_to_end < 1e-12) break;
        
        // Clamp step to not overshoot
        if (std::abs(h) > dist_to_end) {
            h = direction * dist_to_end;
        }
        
        // RKF45 step
        get_vel(t, y, k1);
        
        for (int i = 0; i < dim; ++i) y_temp[i] = y[i] + h * (a21 * k1[i]);
        get_vel(t + c2 * h, y_temp, k2);
        
        for (int i = 0; i < dim; ++i) y_temp[i] = y[i] + h * (a31 * k1[i] + a32 * k2[i]);
        get_vel(t + c3 * h, y_temp, k3);
        
        for (int i = 0; i < dim; ++i) y_temp[i] = y[i] + h * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]);
        get_vel(t + c4 * h, y_temp, k4);
        
        for (int i = 0; i < dim; ++i) y_temp[i] = y[i] + h * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]);
        get_vel(t + c5 * h, y_temp, k5);
        
        for (int i = 0; i < dim; ++i) y_temp[i] = y[i] + h * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i]);
        get_vel(t + c6 * h, y_temp, k6);
        
        // Error estimate
        double max_err = 0.0;
        for (int i = 0; i < dim; ++i) {
            double diff = h * ((b1 - bs1) * k1[i] + (b3 - bs3) * k3[i] + (b4 - bs4) * k4[i] + (b6 - bs6) * k6[i] - bs5 * k5[i]);
            double scale = atol + rtol * std::max(std::abs(y[i]), std::abs(y_temp[i]));
            double ratio = std::abs(diff) / scale;
            if (ratio > max_err) max_err = ratio;
        }
        
        // Accept or reject
        if (max_err <= 1.0) {
            // ACCEPT
            double h_abs = std::abs(h);
            t += h;
            
            // Update state (5th order solution)
            for (int i = 0; i < dim; ++i) {
                y[i] += h * (b1 * k1[i] + b3 * k3[i] + b4 * k4[i] + b6 * k6[i]);
            }
            
            // Accumulate LD (velocity norm * dt)
            get_vel(t, y, vel);
            ld_integral += velocity_norm_lp(vel, ld_p_norm) * h_abs;
            
            // Check if finished
            if ((direction > 0 && t >= t_end) || (direction < 0 && t <= t_end)) {
                break;
            }
        }
        
        // Adjust step size
        double scale_factor;
        if (max_err == 0.0) scale_factor = MAX_SCALE;
        else scale_factor = SAFETY * std::pow(max_err, -0.2);
        scale_factor = std::max(MIN_SCALE, std::min(MAX_SCALE, scale_factor));
        h *= scale_factor;
        
        // Prevent h from becoming too small
        if (std::abs(h) < 1e-15) break;
    }
    
    LDResult result;
    result.ld_value = ld_integral;
    result.final_state = y;
    return result;
}

#endif


