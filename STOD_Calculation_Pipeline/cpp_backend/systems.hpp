#ifndef SYSTEMS_HPP
#define SYSTEMS_HPP

#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <stdexcept>
#include <iostream>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Helper to safely get params
double get_p(const std::map<std::string, double>& p, const std::string& key) {
    if (p.find(key) == p.end()) {
         // Some systems like LiCN don't need params, so we might return 0 if not strict
         return 0.0; 
    }
    return p.at(key);
}

// ---------------------------------------------------------
// BASE CLASS
// ---------------------------------------------------------
class DynamicalSystem {
public:
    virtual ~DynamicalSystem() = default;
    
    // Number of dimensions (2 for Gyre, 3 for Lorenz, 4 for LiCN)
    virtual int get_dim() const = 0;

    // Calculate d(state)/dt
    virtual void get_velocity(double t, const std::vector<double>& state, std::vector<double>& out) const = 0;
    
    // Calculate orthogonal velocity: rotates 2D velocity 90° counterclockwise
    // For 2D: (u, v) -> (-v, u)
    // For higher dims: only rotates first two components, keeps rest unchanged
    virtual void get_velocity_orthogonal(double t, const std::vector<double>& state, std::vector<double>& out) const {
        get_velocity(t, state, out);
        int dim = get_dim();
        if (dim >= 2) {
            double u = out[0];
            double v = out[1];
            out[0] = -v;
            out[1] = u;
        }
    }

    // Calculate the Jacobian matrix J[i][j] = d(v_i)/d(x_j)
    // Output: out is a dim x dim matrix stored row-major (out[i*dim + j] = J[i][j])
    virtual void get_jacobian(double t, const std::vector<double>& state, std::vector<double>& out) const = 0;
    
    // Calculate Jacobian for orthogonal field
    // For 2D: if J = [[a,b],[c,d]] for (u,v), then J_ortho = [[-c,-d],[a,b]] for (-v,u)
    virtual void get_jacobian_orthogonal(double t, const std::vector<double>& state, std::vector<double>& out) const {
        get_jacobian(t, state, out);
        int dim = get_dim();
        if (dim >= 2) {
            // Original: J = [[J00, J01], [J10, J11], ...]
            // Orthogonal field is (-v, u), so:
            //   d(-v)/dx = -J10, d(-v)/dy = -J11
            //   d(u)/dx = J00, d(u)/dy = J01
            double J00 = out[0], J01 = out[1];
            double J10 = out[dim], J11 = out[dim + 1];
            out[0] = -J10;    // d(-v)/dx
            out[1] = -J11;    // d(-v)/dy
            out[dim] = J00;   // d(u)/dx
            out[dim + 1] = J01; // d(u)/dy
        }
    }

    // Map the 2D analysis grid (u, v) -> N-dim Initial State
    // Returns true if valid, false if forbidden (e.g. LiCN energy barrier)
    virtual bool get_initial_state(double u, double v, std::vector<double>& out) const = 0;

    // Map N-dim State -> 2D Analysis Grid (x, y) for SOD/FTLE
    virtual void project_to_grid(const std::vector<double>& state, double& x, double& y) const {
        // Default: take first two dimensions
        x = state[0];
        y = state[1];
    }
};

// ---------------------------------------------------------
// 1. DOUBLE GYRE (2D)
// ---------------------------------------------------------
class DoubleGyre : public DynamicalSystem {
    double A, eps, omega;
public:
    DoubleGyre(const std::map<std::string, double>& params) {
        A = get_p(params, "A");
        eps = get_p(params, "epsilon");
        omega = get_p(params, "omega");
    }
    int get_dim() const override { return 2; }
    
    bool get_initial_state(double u, double v, std::vector<double>& out) const override {
        out = {u, v};
        return true;
    }

    void get_velocity(double t, const std::vector<double>& s, std::vector<double>& out) const override {
        double x = s[0], y = s[1];
        double a_t = eps * std::sin(omega * t);
        double b_t = 1.0 - 2.0 * a_t;
        double f = a_t * x * x + b_t * x;
        double dfdx = 2.0 * a_t * x + b_t;
        out[0] = -M_PI * A * std::sin(M_PI * f) * std::cos(M_PI * y);
        out[1] =  M_PI * A * std::cos(M_PI * f) * std::sin(M_PI * y) * dfdx;
    }

    void get_jacobian(double t, const std::vector<double>& s, std::vector<double>& out) const override {
        // J[i][j] = d(v_i)/d(x_j), stored row-major: out[i*dim + j]
        double x = s[0], y = s[1];
        double a_t = eps * std::sin(omega * t);
        double b_t = 1.0 - 2.0 * a_t;
        double f = a_t * x * x + b_t * x;
        double dfdx = 2.0 * a_t * x + b_t;
        double d2fdx2 = 2.0 * a_t;
        
        double sin_pf = std::sin(M_PI * f);
        double cos_pf = std::cos(M_PI * f);
        double sin_py = std::sin(M_PI * y);
        double cos_py = std::cos(M_PI * y);
        
        // v0 = -πA sin(πf) cos(πy)
        // dv0/dx = -π²A cos(πf) cos(πy) * dfdx
        // dv0/dy =  π²A sin(πf) sin(πy)
        out[0] = -M_PI * M_PI * A * cos_pf * cos_py * dfdx;       // J[0][0]
        out[1] =  M_PI * M_PI * A * sin_pf * sin_py;              // J[0][1]
        
        // v1 = πA cos(πf) sin(πy) * dfdx
        // dv1/dx = -π²A sin(πf) sin(πy) * dfdx² + πA cos(πf) sin(πy) * d2fdx2
        // dv1/dy = π²A cos(πf) cos(πy) * dfdx
        out[2] = -M_PI * M_PI * A * sin_pf * sin_py * dfdx * dfdx 
                 + M_PI * A * cos_pf * sin_py * d2fdx2;           // J[1][0]
        out[3] =  M_PI * M_PI * A * cos_pf * cos_py * dfdx;       // J[1][1]
    }
};

// ---------------------------------------------------------
// 2. LORENZ '63 (Full 3D System)
// ---------------------------------------------------------
// The classic Lorenz '63 chaotic attractor.
// This is a 3D system where z evolves dynamically.
// The 2D analysis grid is (x, y) starting from a fixed z0 plane.
//
// Equations:
//   dx/dt = σ(y - x)
//   dy/dt = x(ρ - z) - y
//   dz/dt = xy - βz
//
// Note: For orthogonal mode, only the first two velocity components are rotated.
class Lorenz : public DynamicalSystem {
    double sigma, rho, beta, z0;
public:
    Lorenz(const std::map<std::string, double>& params) {
        sigma = get_p(params, "sigma");
        rho   = get_p(params, "rho");
        beta  = get_p(params, "beta");
        // z0 is the initial z-plane for starting trajectories
        if (params.find("initial_z0") != params.end()) z0 = params.at("initial_z0");
        else z0 = 27.0; // Default: z0 = 27 (near the Lorenz attractor)
    }
    
    // Full 3D Lorenz system
    int get_dim() const override { return 3; }

    bool get_initial_state(double u, double v, std::vector<double>& out) const override {
        // Grid is (x, y), trajectories start at z = z0
        out = {u, v, z0};
        return true;
    }

    void project_to_grid(const std::vector<double>& s, double& x, double& y) const override {
        // Project 3D state to 2D analysis grid (x, y)
        x = s[0];
        y = s[1];
    }

    void get_velocity(double, const std::vector<double>& s, std::vector<double>& out) const override {
        // Full 3D Lorenz equations
        double x = s[0], y = s[1], z = s[2];
        out[0] = sigma * (y - x);       // dx/dt = σ(y - x)
        out[1] = x * (rho - z) - y;     // dy/dt = x(ρ - z) - y
        out[2] = x * y - beta * z;      // dz/dt = xy - βz
    }

    void get_jacobian(double, const std::vector<double>& s, std::vector<double>& out) const override {
        // 3D Jacobian for full Lorenz system
        // dx/dt = σ(y-x), dy/dt = x(ρ-z)-y, dz/dt = xy - βz
        double x = s[0], y = s[1], z = s[2];
        
        // Row 0: dv0/dx = -σ, dv0/dy = σ, dv0/dz = 0
        out[0] = -sigma;  out[1] = sigma;   out[2] = 0.0;
        // Row 1: dv1/dx = ρ-z, dv1/dy = -1, dv1/dz = -x
        out[3] = rho - z; out[4] = -1.0;    out[5] = -x;
        // Row 2: dv2/dx = y, dv2/dy = x, dv2/dz = -β
        out[6] = y;       out[7] = x;       out[8] = -beta;
    }
};

// ---------------------------------------------------------
// 3. LiCN ISOMERIZATION (4D)
// ---------------------------------------------------------
// Implementing the Legendre Expansion and Hamiltonian with ANALYTICAL Jacobian
// Ported from Fortran D2POT subroutine for efficient FLI/variational calculations
class LiCN : public DynamicalSystem {
    // Constants hardcoded to match Python and Fortran
    const double CTE = 1822.83;
    const double M_C = 12.0 * 1822.83;
    const double M_N = 14.003074 * 1822.83;
    const double M_LI = 7.0160045 * 1822.83;
    const double R_E_CN = 2.186;
    double MU1, MU2;
    double target_energy;
    
    // Potential coefficients (from Fortran D2POT)
    static constexpr double ALPHA_DAMP = 1.515625;
    static constexpr double R0_DAMP = 1.900781;
    static constexpr double RSHORT = 9.0;
    static constexpr double V0_OFFSET = 0.24460547;
    
    // Coefficients A, B, C for short-range exponential terms
    const double A[10] = {0.1383211600E01, 0.2957913200E01, 0.4742029700E01, 0.1888529900E01, 
                          0.4414335400E01, 0.4025649600E01, 0.5842589900E01, 0.2616811400E01, 
                          0.6344657900E01, -0.1520228000E02};
    const double B[10] = {-0.1400070600E00, -0.1479771600E01, -0.1811986200E01, -0.1287503000E01, 
                          -0.2322971400E01, -0.2775383200E01, -0.3480852900E01, -0.2655595200E01, 
                          -0.4344980100E01, 0.6549253700E01};
    const double C_coef[10] = {-0.2078921600E00, 0.1161308200E-01, 0.171806800E-01, -0.2772849100E-01, 
                               0.7069274200E-01, 0.1377197800E00, 0.1863111400E00, 0.5881550400E-02, 
                               0.1529136800E00, -0.1302567800E01};
    const double D_coef[7] = {-0.10E01, -0.215135E00, -0.3414573000E01, -0.3818815000E01, 
                              -0.1584152000E02, -0.1429374000E02, -0.4381719000E02};
    
    // Long-range dispersion coefficients (from Fortran)
    static constexpr double C40 = -10.5271, C42 = -3.17;
    static constexpr double C51 = -20.62328, C53 = 3.73208;
    static constexpr double C60 = -57.49396, C62 = -106.8192, C64 = 17.14139;
    static constexpr double C71 = -202.8972, C73 = -75.23207, C75 = -28.45514;
    static constexpr double C80 = -458.2015, C82 = -353.7347, C84 = -112.6427, C86 = -108.2786;

public:
    LiCN(const std::map<std::string, double>& params) {
        MU1 = (M_LI * (M_C + M_N)) / (M_LI + M_C + M_N);
        MU2 = (M_C * M_N) / (M_C + M_N);
        if (params.find("target_energy_au") != params.end()) 
            target_energy = params.at("target_energy_au");
        else 
            target_energy = 0.01822534;
    }
    int get_dim() const override { return 4; } // R, theta, P_R, P_theta

    // --- Legendre Polynomials and Derivatives ---
    // Computes P[0..N], dP[0..N] (derivative w.r.t. x = cos(theta))
    void compute_legendre(double x, int N, double* P, double* dP) const {
        P[0] = 1.0;
        dP[0] = 0.0;
        if (N >= 1) {
            P[1] = x;
            dP[1] = 1.0;
        }
        for (int L = 2; L <= N; ++L) {
            P[L] = ((2*L - 1) * x * P[L-1] - (L-1) * P[L-2]) / L;
            dP[L] = L * P[L-1] + x * dP[L-1];
        }
    }
    
    // Computes P, dP, d2P (second derivative w.r.t. x)
    void compute_legendre_d2(double x, int N, double* P, double* dP, double* d2P) const {
        compute_legendre(x, N, P, dP);
        d2P[0] = 0.0;
        if (N >= 1) d2P[1] = 0.0;
        for (int L = 2; L <= N; ++L) {
            d2P[L] = (L + 1) * dP[L-1] + x * d2P[L-1];
        }
    }

    // --- ANALYTICAL Potential with First and Second Derivatives ---
    // Ported from Fortran D2POT subroutine
    void get_potential_analytical(double R, double theta, 
                                   double& V, double& dVdR, double& dVdTh,
                                   double& d2VdR2, double& d2VdTh2, double& d2VdRdTh) const {
        // Compute tempered inverse powers of R and their derivatives
        double DR = R - R0_DAMP;
        double AUX0 = std::exp(-ALPHA_DAMP * DR * DR);
        
        double RINV[8], DRINV[8], D2RINV[8];
        for (int K = 0; K < 8; ++K) {
            int k = K + 1;  // 1-indexed power
            double R_pow_k = std::pow(R, k);
            RINV[K] = (1.0 - AUX0) / R_pow_k;
            DRINV[K] = 2.0 * ALPHA_DAMP * DR * AUX0 / R_pow_k - k * RINV[K] / R;
            D2RINV[K] = (RINV[K] / R - DRINV[K]) * k / R 
                       + (R * (1.0 - 2.0 * ALPHA_DAMP * DR * DR) - k * DR) 
                         * 2.0 * ALPHA_DAMP * AUX0 / std::pow(R, k + 1);
        }
        
        // Compute VL coefficients and their R-derivatives (indices 0-9, but VL[7,8,9] are special)
        double VL[10], DVL[10], D2VL[10];
        
        // Long-range contributions (tempered)
        VL[0] = C40*RINV[3] + C60*RINV[5] + C80*RINV[7] + RINV[0]*D_coef[0];
        VL[1] = C51*RINV[4] + C71*RINV[6] + RINV[1]*D_coef[1];
        VL[2] = C42*RINV[3] + C62*RINV[5] + C82*RINV[7] + RINV[2]*D_coef[2];
        VL[3] = C53*RINV[4] + C73*RINV[6] + RINV[3]*D_coef[3];
        VL[4] = C64*RINV[5] + C84*RINV[7] + RINV[4]*D_coef[4];
        VL[5] = C75*RINV[6] + RINV[5]*D_coef[5];
        VL[6] = C86*RINV[7] + RINV[6]*D_coef[6];
        VL[7] = 0.0; VL[8] = 0.0; VL[9] = 0.0;
        
        DVL[0] = C40*DRINV[3] + C60*DRINV[5] + C80*DRINV[7] + DRINV[0]*D_coef[0];
        DVL[1] = C51*DRINV[4] + C71*DRINV[6] + DRINV[1]*D_coef[1];
        DVL[2] = C42*DRINV[3] + C62*DRINV[5] + C82*DRINV[7] + DRINV[2]*D_coef[2];
        DVL[3] = C53*DRINV[4] + C73*DRINV[6] + DRINV[3]*D_coef[3];
        DVL[4] = C64*DRINV[5] + C84*DRINV[7] + DRINV[4]*D_coef[4];
        DVL[5] = C75*DRINV[6] + DRINV[5]*D_coef[5];
        DVL[6] = C86*DRINV[7] + DRINV[6]*D_coef[6];
        DVL[7] = 0.0; DVL[8] = 0.0; DVL[9] = 0.0;
        
        D2VL[0] = C40*D2RINV[3] + C60*D2RINV[5] + C80*D2RINV[7] + D2RINV[0]*D_coef[0];
        D2VL[1] = C51*D2RINV[4] + C71*D2RINV[6] + D2RINV[1]*D_coef[1];
        D2VL[2] = C42*D2RINV[3] + C62*D2RINV[5] + C82*D2RINV[7] + D2RINV[2]*D_coef[2];
        D2VL[3] = C53*D2RINV[4] + C73*D2RINV[6] + D2RINV[3]*D_coef[3];
        D2VL[4] = C64*D2RINV[5] + C84*D2RINV[7] + D2RINV[4]*D_coef[4];
        D2VL[5] = C75*D2RINV[6] + D2RINV[5]*D_coef[5];
        D2VL[6] = C86*D2RINV[7] + D2RINV[6]*D_coef[6];
        D2VL[7] = 0.0; D2VL[8] = 0.0; D2VL[9] = 0.0;
        
        // Short-range exponential corrections
        if (R <= RSHORT) {
            double R2 = R * R;
            for (int L = 0; L < 10; ++L) {
                double DL_val = std::exp(A[L] + B[L]*R + C_coef[L]*R2);
                double DDL_val = (B[L] + 2.0*C_coef[L]*R) * DL_val;
                double D2DL_val = 2.0*C_coef[L]*DL_val + (B[L] + 2.0*C_coef[L]*R)*DDL_val;
                VL[L] += DL_val;
                DVL[L] += DDL_val;
                D2VL[L] += D2DL_val;
            }
        }
        
        // Compute Legendre polynomials and derivatives w.r.t. cos(theta)
        double cos_th = std::cos(theta);
        double sin_th = std::sin(theta);
        double P[10], dP_dx[10], d2P_dx2[10];
        compute_legendre_d2(cos_th, 9, P, dP_dx, d2P_dx2);
        
        // Convert derivatives from d/d(cos θ) to d/dθ
        // dP/dθ = dP/dx * dx/dθ = -sin(θ) * dP/dx
        // d²P/dθ² = d/dθ(-sin(θ) * dP/dx) = -cos(θ)*dP/dx + sin²(θ)*d²P/dx²
        double dP_dth[10], d2P_dth2[10];
        for (int L = 0; L <= 9; ++L) {
            dP_dth[L] = -sin_th * dP_dx[L];
            d2P_dth2[L] = -cos_th * dP_dx[L] + sin_th * sin_th * d2P_dx2[L];
        }
        
        // Sum up potential and derivatives
        V = VL[0] * P[0];  // V0 term (L=0)
        dVdR = DVL[0] * P[0];
        dVdTh = VL[0] * dP_dth[0];
        d2VdR2 = D2VL[0] * P[0];
        d2VdTh2 = VL[0] * d2P_dth2[0];
        d2VdRdTh = DVL[0] * dP_dth[0];
        
        for (int L = 1; L <= 9; ++L) {
            V += VL[L] * P[L];
            dVdR += DVL[L] * P[L];
            dVdTh += VL[L] * dP_dth[L];
            d2VdR2 += D2VL[L] * P[L];
            d2VdTh2 += VL[L] * d2P_dth2[L];
            d2VdRdTh += DVL[L] * dP_dth[L];
        }
        
        V += V0_OFFSET;
    }
    
    // --- Simplified potential getter (for get_velocity) ---
    void get_potential(double R, double theta, double& V_val, double& dVdR, double& dVdTh) const {
        double d2VdR2, d2VdTh2, d2VdRdTh;
        get_potential_analytical(R, theta, V_val, dVdR, dVdTh, d2VdR2, d2VdTh2, d2VdRdTh);
    }

    // --- State Reconstruction (Grid -> 4D) ---
    bool get_initial_state(double psi, double p_psi, std::vector<double>& out) const override {
        // grid u = psi (theta), v = p_psi
        double theta = psi;
        
        // MEP R_eq(theta)
        double cos_t = std::cos(theta);
        double R_eq = 4.1159 + 0.2551*cos_t + 0.49830*std::cos(2*theta) - 0.0534270*std::cos(3*theta) 
                      - 0.0681241*std::cos(4*theta) + 0.0205782*std::cos(5*theta);
        
        double dR_eq_dtheta = -0.2551*std::sin(theta) - 2.0*0.49830*std::sin(2*theta) + 3.0*0.0534270*std::sin(3*theta) 
                              + 4.0*0.0681241*std::sin(4*theta) - 5.0*0.0205782*std::sin(5*theta);
        
        double R = R_eq;
        double V_val, dVdR, dVdTh;
        get_potential(R, theta, V_val, dVdR, dVdTh);

        if (V_val > target_energy) return false;

        // Inertia
        double term1 = 1.0 / (MU1 * R*R);
        double term2 = 1.0 / (MU2 * R_E_CN*R_E_CN);
        double I_val = 1.0 / (term1 + term2);

        // Solve for P_R
        double inv_2mu1 = 1.0 / (2.0 * MU1);
        double inv_2I = 1.0 / (2.0 * I_val);
        double Rp = dR_eq_dtheta;

        double A_q = inv_2mu1 + (Rp*Rp * inv_2I);
        double B_q = (2.0 * p_psi * Rp) * inv_2I;
        double C_q = (p_psi*p_psi * inv_2I) + V_val - target_energy;

        double disc = B_q*B_q - 4*A_q*C_q;
        if (disc < 0) return false;

        double sqrt_D = std::sqrt(disc);
        double P_R_1 = (-B_q + sqrt_D) / (2*A_q);
        double P_R_2 = (-B_q - sqrt_D) / (2*A_q);

        // Branch selection: P_theta > 0
        double P_th_1 = p_psi + Rp * P_R_1;
        double P_th_2 = p_psi + Rp * P_R_2;

        double P_R, P_theta;
        if (P_th_1 > 0) { P_R = P_R_1; P_theta = P_th_1; }
        else { P_R = P_R_2; P_theta = P_th_2; }

        out = {R, theta, P_R, P_theta};
        return true;
    }

    // --- Projection Back to Grid ---
    void project_to_grid(const std::vector<double>& s, double& x, double& y) const override {
        // s = [R, theta, P_R, P_theta]
        // x = theta
        // y = P_psi = P_theta - R'(theta)*P_R
        double theta = s[1];
        double P_R = s[2];
        double P_theta = s[3];

        double dR_eq_dtheta = -0.2551*std::sin(theta) - 2.0*0.49830*std::sin(2*theta) + 3.0*0.0534270*std::sin(3*theta) 
                              + 4.0*0.0681241*std::sin(4*theta) - 5.0*0.0205782*std::sin(5*theta);
        
        x = theta;
        y = P_theta - (dR_eq_dtheta * P_R);
    }

    // --- EOM ---
    void get_velocity(double, const std::vector<double>& s, std::vector<double>& out) const override {
        double R = s[0], theta = s[1], P_R = s[2], P_theta = s[3];
        
        double term1 = 1.0 / (MU1 * R*R);
        double term2 = 1.0 / (MU2 * R_E_CN*R_E_CN);
        double I_th = 1.0 / (term1 + term2);

        double V_val, dVdR, dVdTh;
        get_potential(R, theta, V_val, dVdR, dVdTh);

        double dT_bend_dR = (P_theta*P_theta / 2.0) * (-2.0 / (MU1 * R*R*R));

        out[0] = P_R / MU1;
        out[1] = P_theta / I_th;
        out[2] = -dVdR - dT_bend_dR;
        out[3] = -dVdTh;
    }

    void get_jacobian(double, const std::vector<double>& s, std::vector<double>& out) const override {
        // State: [R, theta, P_R, P_theta]
        // EOM:
        //   dR/dt     = P_R / MU1
        //   dtheta/dt = P_theta / I_th  where I_th = 1/(1/(MU1*R²) + 1/(MU2*r_CN²))
        //   dP_R/dt   = -dV/dR + P_theta²/(MU1*R³)
        //   dP_theta/dt = -dV/dtheta
        //
        // ANALYTICAL JACOBIAN - much faster than numerical differentiation!
        
        double R = s[0], theta = s[1], P_theta = s[3];
        double R2 = R * R;
        double R3 = R2 * R;
        double R4 = R3 * R;
        
        double term1 = 1.0 / (MU1 * R2);
        double term2 = 1.0 / (MU2 * R_E_CN * R_E_CN);
        double I_th = 1.0 / (term1 + term2);
        
        // dI_th/dR = 2*I_th²/(MU1*R³)
        double dI_th_dR = 2.0 * I_th * I_th / (MU1 * R3);
        
        // Get ANALYTICAL second derivatives of potential
        double V_val, dVdR, dVdTh, d2VdR2, d2VdTh2, d2VdRdTh;
        get_potential_analytical(R, theta, V_val, dVdR, dVdTh, d2VdR2, d2VdTh2, d2VdRdTh);
        
        // Initialize output (4x4 = 16 elements)
        for (int i = 0; i < 16; ++i) out[i] = 0.0;
        
        // Row 0: d(dR/dt)/d(...) = d(P_R/MU1)/d(...)
        // Only non-zero: d/dP_R = 1/MU1
        out[0*4 + 2] = 1.0 / MU1;  // J[0][2]
        
        // Row 1: d(dtheta/dt)/d(...) = d(P_theta/I_th)/d(...)
        // d/dR = -P_theta * dI_th_dR / I_th²
        // d/dP_theta = 1/I_th
        out[1*4 + 0] = -P_theta * dI_th_dR / (I_th * I_th);  // J[1][0]
        out[1*4 + 3] = 1.0 / I_th;                           // J[1][3]
        
        // Row 2: d(dP_R/dt)/d(...) = d(-dV/dR + P_theta²/(MU1*R³))/d(...)
        // d/dR = -d²V/dR² - 3*P_theta²/(MU1*R⁴)
        // d/dtheta = -d²V/dRdtheta
        // d/dP_theta = 2*P_theta/(MU1*R³)
        out[2*4 + 0] = -d2VdR2 - 3.0 * P_theta * P_theta / (MU1 * R4);  // J[2][0]
        out[2*4 + 1] = -d2VdRdTh;                                        // J[2][1]
        out[2*4 + 3] = 2.0 * P_theta / (MU1 * R3);                       // J[2][3]
        
        // Row 3: d(dP_theta/dt)/d(...) = d(-dV/dtheta)/d(...)
        // d/dR = -d²V/dRdtheta
        // d/dtheta = -d²V/dtheta²
        out[3*4 + 0] = -d2VdRdTh;  // J[3][0]
        out[3*4 + 1] = -d2VdTh2;   // J[3][1]
    }
};

// ---------------------------------------------------------
// STANDARD SYSTEMS
// ---------------------------------------------------------
class Duffing : public DynamicalSystem {
    double alpha, beta, delta, gamma, omega;
public:
    Duffing(const std::map<std::string, double>& params) {
        alpha = get_p(params, "alpha"); beta  = get_p(params, "beta");
        delta = get_p(params, "delta"); gamma = get_p(params, "gamma");
        omega = get_p(params, "omega");
    }
    int get_dim() const override { return 2; }
    bool get_initial_state(double u, double v, std::vector<double>& out) const override { out = {u, v}; return true; }
    void get_velocity(double t, const std::vector<double>& s, std::vector<double>& out) const override {
        out[0] = s[1];
        out[1] = gamma * std::cos(omega * t) - delta * s[1] - alpha * s[0] - beta * std::pow(s[0], 3);
    }
    void get_jacobian(double, const std::vector<double>& s, std::vector<double>& out) const override {
        // Duffing: dx/dt = y, dy/dt = γcos(ωt) - δy - αx - βx³
        double x = s[0];
        // Row 0: dv0/dx = 0, dv0/dy = 1
        out[0] = 0.0;                              // J[0][0]
        out[1] = 1.0;                              // J[0][1]
        // Row 1: dv1/dx = -α - 3βx², dv1/dy = -δ
        out[2] = -alpha - 3.0 * beta * x * x;      // J[1][0]
        out[3] = -delta;                           // J[1][1]
    }
};

class NonlinearSaddle : public DynamicalSystem {
    double alpha, beta, gamma;
public:
    NonlinearSaddle(const std::map<std::string, double>& params) {
        alpha = get_p(params, "alpha"); beta = get_p(params, "beta"); gamma = get_p(params, "gamma");
    }
    int get_dim() const override { return 2; }
    bool get_initial_state(double u, double v, std::vector<double>& out) const override { out = {u, v}; return true; }
    void get_velocity(double, const std::vector<double>& s, std::vector<double>& out) const override {
        out[0] =  alpha * s[0] + beta * s[0] * s[1];
        out[1] = -alpha * s[1] + gamma * s[0] * s[0];
    }
    void get_jacobian(double, const std::vector<double>& s, std::vector<double>& out) const override {
        // NonlinearSaddle: dx/dt = αx + βxy, dy/dt = -αy + γx²
        double x = s[0], y = s[1];
        // Row 0: dv0/dx = α + βy, dv0/dy = βx
        out[0] = alpha + beta * y;    // J[0][0]
        out[1] = beta * x;            // J[0][1]
        // Row 1: dv1/dx = 2γx, dv1/dy = -α
        out[2] = 2.0 * gamma * x;     // J[1][0]
        out[3] = -alpha;              // J[1][1]
    }
};

class HyperbolicLinear : public DynamicalSystem {
    double alpha;
public:
    HyperbolicLinear(const std::map<std::string, double>& params) { alpha = get_p(params, "alpha"); }
    int get_dim() const override { return 2; }
    bool get_initial_state(double u, double v, std::vector<double>& out) const override { out = {u, v}; return true; }
    void get_velocity(double, const std::vector<double>& s, std::vector<double>& out) const override {
        out[0] =  alpha * s[0];
        out[1] = -alpha * s[1];
    }
    void get_jacobian(double, const std::vector<double>& s, std::vector<double>& out) const override {
        // HyperbolicLinear: dx/dt = αx, dy/dt = -αy
        // Jacobian is constant:
        out[0] =  alpha;  // J[0][0]
        out[1] =  0.0;    // J[0][1]
        out[2] =  0.0;    // J[1][0]
        out[3] = -alpha;  // J[1][1]
    }
};

// ---------------------------------------------------------
// SIMPLE PENDULUM (2D Phase Space)
// ---------------------------------------------------------
// Equations of motion:
//   dtheta/dt = omega
//   domega/dt = -lambda * sin(theta)
//
// Key features:
//   - Elliptic fixed point at (0, 0) - stable equilibrium
//   - Hyperbolic saddle at (±π, 0) - unstable equilibrium
//   - Separatrix connects the saddles (homoclinic orbit)
class Pendulum : public DynamicalSystem {
    double lambda;  // g/L ratio
public:
    Pendulum(const std::map<std::string, double>& params) {
        lambda = get_p(params, "lambda");
        if (lambda == 0.0) lambda = 1.0;  // Default to canonical form
    }
    int get_dim() const override { return 2; }
    
    bool get_initial_state(double u, double v, std::vector<double>& out) const override {
        // u = theta (angle), v = omega (angular velocity)
        out = {u, v};
        return true;
    }
    
    void get_velocity(double, const std::vector<double>& s, std::vector<double>& out) const override {
        // s[0] = theta, s[1] = omega
        double theta = s[0], omega = s[1];
        out[0] = omega;                      // dtheta/dt = omega
        out[1] = -lambda * std::sin(theta);  // domega/dt = -lambda * sin(theta)
    }
    
    void get_jacobian(double, const std::vector<double>& s, std::vector<double>& out) const override {
        // Pendulum: dtheta/dt = omega, domega/dt = -lambda * sin(theta)
        // Jacobian:
        //   d(dtheta/dt)/dtheta = 0,       d(dtheta/dt)/domega = 1
        //   d(domega/dt)/dtheta = -lambda*cos(theta), d(domega/dt)/domega = 0
        double theta = s[0];
        out[0] = 0.0;                         // J[0][0]
        out[1] = 1.0;                         // J[0][1]
        out[2] = -lambda * std::cos(theta);   // J[1][0]
        out[3] = 0.0;                         // J[1][1]
    }
};

// ---------------------------------------------------------
// FACTORY
// ---------------------------------------------------------
DynamicalSystem* create_system(const std::string& name, const std::map<std::string, double>& params) {
    if (name == "doublegyre")        return new DoubleGyre(params);
    if (name == "duffing")           return new Duffing(params);
    if (name == "nonlinear_saddle")  return new NonlinearSaddle(params);
    if (name == "hyperbolic_linear") return new HyperbolicLinear(params);
    if (name == "lorenz")            return new Lorenz(params);
    if (name == "licn")              return new LiCN(params);
    if (name == "pendulum")          return new Pendulum(params);
    return nullptr;
}

#endif


