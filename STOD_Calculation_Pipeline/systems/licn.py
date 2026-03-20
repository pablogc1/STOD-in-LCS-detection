# ----------------------------------------------------------------------
#        SYSTEM DEFINITION: LiCN Isomerization (2 DOF Reduced)
# ----------------------------------------------------------------------
# Based on the Hamiltonian provided in Garcia-Cuadrillero et al.
# and the "DEFINITIVE SCRIPT" for the potential surface.
# ----------------------------------------------------------------------

import numpy as np
from numpy.polynomial.legendre import legval

# --- PHYSICAL CONSTANTS (from plotting script) ---
CTE = 1822.83
M_C = 12.0 * CTE
M_N = 14.003074 * CTE
M_LI = 7.0160045 * CTE

# Reduced masses
MU1 = (M_LI * (M_C + M_N)) / (M_LI + M_C + M_N)
MU2 = (M_C * M_N) / (M_C + M_N)

# Equilibrium distance for C-N (rigid rod approximation in 2D model)
R_E_CN = 2.186

# --- POTENTIAL ENERGY SURFACE IMPLEMENTATION ---
def _pot_subroutine(R):
    A = [0.1383211600E01, 0.2957913200E01, 0.4742029700E01, 0.1888529900E01, 0.4414335400E01,
         0.4025649600E01, 0.5842589900E01, 0.2616811400E01, 0.6344657900E01, -0.1520228000E02]
    B = [-0.1400070600E00, -0.1479771600E01, -0.1811986200E01, -0.1287503000E01, -0.2322971400E01,
         -0.2775383200E01, -0.3480852900E01, -0.2655595200E01, -0.4344980100E01, 0.6549253700E01]
    C = [-0.2078921600E00, 0.1161308200E-01, 0.171806800E-01, -0.2772849100E-01, 0.7069274200E-01,
         0.1377197800E00, 0.1863111400E00, 0.5881550400E-02, 0.1529136800E00, -0.1302567800E01]
    D = [-0.10E01, -0.215135E00, -0.3414573000E01, -0.3818815000E01, -0.1584152000E02,
         -0.1429374000E02, -0.4381719000E02]
    
    C40, C42 = -1.05271E1, -3.17E0
    C51, C53 = -2.062328E1, 3.7320800E0
    C60, C62, C64 = -5.749396E1, -1.068192E2, 1.714139E1
    C71, C73, C75 = -2.028972E2, -7.523207E1, -2.845514E1
    C80, C82, C84, C86 = -4.582015E2, -3.537347E2, -1.126427E2, -1.126427E2
    
    DEMPA, DEMPRO, RSHORT = -1.515625E0, 1.900781E0, 9.0
    R2 = R*R
    RINV = np.zeros(8)
    VL = np.zeros(9)
    DR = R - DEMPRO
    DAMPOT = 1.0 - np.exp(DEMPA * DR * DR)
    RINV[0] = DAMPOT / R
    
    for L in range(7):
        RINV[L+1] = RINV[L] / R
        
    V0 = C40*RINV[3] + C60*RINV[5] + C80*RINV[7] + RINV[0]*D[0]
    VL[0] = C51*RINV[4] + C71*RINV[6] + RINV[1]*D[1]
    VL[1] = C42*RINV[3] + C62*RINV[5] + C82*RINV[7] + RINV[2]*D[2]
    VL[2] = C53*RINV[4] + C73*RINV[6] + RINV[3]*D[3]
    VL[3] = C64*RINV[5] + C84*RINV[7] + RINV[4]*D[4]
    VL[4] = C75*RINV[6] + RINV[5]*D[5]
    VL[5] = C86*RINV[7] + RINV[6]*D[6]
    
    if R <= RSHORT:
        V0 += np.exp(A[0] + B[0]*R + C[0]*R2)
        for i in range(6):
            VL[i] += np.exp(A[i+1] + B[i+1]*R + C[i+1]*R2)
        for i in range(6, 9):
            VL[i] = np.exp(A[i+1] + B[i+1]*R + C[i+1]*R2)
            
    return V0, VL

def _get_potential(R, theta):
    """Calculates V(R, theta) using Legendre polynomial expansion."""
    V0, VL = _pot_subroutine(R)
    x = np.cos(theta)
    
    val = np.zeros(10)
    for l in range(10):
        coeffs = np.zeros(l+1)
        coeffs[l] = 1
        val[l] = legval(x, coeffs)
        
    # Constant shift from script: +0.24460547
    return V0 + np.sum(VL * val[1:]) + 0.24460547

def _get_potential_gradient(R, theta, eps=1e-8):
    """
    Computes numerical gradient of V using 4th order central difference.
    Analytic differentiation of the subroutine is prone to transcription errors.
    """
    # dV/dR
    v_r_m2 = _get_potential(R - 2*eps, theta)
    v_r_m1 = _get_potential(R - eps, theta)
    v_r_p1 = _get_potential(R + eps, theta)
    v_r_p2 = _get_potential(R + 2*eps, theta)
    dVdR = (v_r_m2 - 8*v_r_m1 + 8*v_r_p1 - v_r_p2) / (12*eps)

    # dV/dtheta
    v_t_m2 = _get_potential(R, theta - 2*eps)
    v_t_m1 = _get_potential(R, theta - eps)
    v_t_p1 = _get_potential(R, theta + eps)
    v_t_p2 = _get_potential(R, theta + 2*eps)
    dVdTh = (v_t_m2 - 8*v_t_m1 + 8*v_t_p1 - v_t_p2) / (12*eps)
    
    return dVdR, dVdTh

def _get_moment_inertia(R):
    """I_theta(R, r_e)"""
    # I = [1/(mu1 R^2) + 1/(mu2 r^2)]^-1
    term1 = 1.0 / (MU1 * R**2)
    term2 = 1.0 / (MU2 * R_E_CN**2)
    return 1.0 / (term1 + term2)

# --- PIPELINE INTERFACE FUNCTIONS ---

def get_velocity_field(state, t, physics_params):
    """
    Returns [dR/dt, dTheta/dt, dP_R/dt, dP_Theta/dt]
    State vector: [R, theta, P_R, P_theta]
    """
    R, theta, P_R, P_theta = state
    
    # 1. Properties
    I_th = _get_moment_inertia(R)
    dVdR, dVdTh = _get_potential_gradient(R, theta)
    
    # 2. Derivatives of Kinetic Energy T w.r.t R
    dT_bend_dR = (P_theta**2 / 2.0) * (-2.0 / (MU1 * R**3))
    
    # 3. Equations of Motion (Hamilton's Eqs)
    dR_dt = P_R / MU1
    dTheta_dt = P_theta / I_th
    dP_R_dt = -dVdR - dT_bend_dR
    dP_Theta_dt = -dVdTh
    
    return np.array([dR_dt, dTheta_dt, dP_R_dt, dP_Theta_dt], dtype=float)

def get_initial_state_from_grid(row, col, grid_params):
    """
    Reconstructs the 4D state from the 2D Poincaré Surface of Section grid.
    Returns None if the state is energetically forbidden.
    """
    # 1. Map Grid to PSOS coordinates (Psi, P_Psi)
    nx = int(grid_params['grid_resolution_x'])
    ny = int(grid_params['grid_resolution_y'])
    
    psi_min, psi_max = grid_params['x_min'], grid_params['x_max']
    ppsi_min, ppsi_max = grid_params['y_min'], grid_params['y_max']
    
    # Center of the cell
    psi = psi_min + (col + 0.5) * (psi_max - psi_min) / nx
    p_psi = ppsi_min + (row + 0.5) * (ppsi_max - ppsi_min) / ny
    
    # 2. Retrieve Physical Params
    E_target = grid_params.get('target_energy_au')
    if E_target is None:
        raise ValueError("target_energy_au must be in grid_params for LiCN PSOS reconstruction")

    # 3. Calculate R_e(theta) and dR_e/dtheta (MEP)
    theta = psi
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    R_eq = (4.1159 + 0.2551*cos_t + 0.49830*np.cos(2*theta) - 0.0534270*np.cos(3*theta) 
            - 0.0681241*np.cos(4*theta) + 0.0205782*np.cos(5*theta))
            
    dR_eq_dtheta = (-0.2551*sin_t - 2.0*0.49830*np.sin(2*theta) + 3.0*0.0534270*np.sin(3*theta) 
                    + 4.0*0.0681241*np.sin(4*theta) - 5.0*0.0205782*np.sin(5*theta))

    # 4. Set R
    R = R_eq
    
    # 5. Solve for P_R using Energy Conservation
    V_val = _get_potential(R, theta)
    I_val = _get_moment_inertia(R)
    
    # Check if classically forbidden (Potential > Energy)
    if V_val > E_target:
        return None

    # Quadratic coefficients for A*P_R^2 + B*P_R + C = 0
    inv_2mu1 = 1.0 / (2.0 * MU1)
    inv_2I = 1.0 / (2.0 * I_val)
    Rp = dR_eq_dtheta
    
    A_quad = inv_2mu1 + (Rp**2 * inv_2I)
    B_quad = (2.0 * p_psi * Rp) * inv_2I
    C_quad = (p_psi**2 * inv_2I) + V_val - E_target
    
    discriminant = B_quad**2 - 4*A_quad*C_quad
    
    # Momentum is imaginary -> Forbidden
    if discriminant < 0:
        return None
        
    sqrt_D = np.sqrt(discriminant)
    P_R_1 = (-B_quad + sqrt_D) / (2*A_quad)
    P_R_2 = (-B_quad - sqrt_D) / (2*A_quad)
    
    # Check condition dot(theta) > 0 => P_theta > 0
    P_theta_1 = p_psi + Rp * P_R_1
    P_theta_2 = p_psi + Rp * P_R_2
    
    if P_theta_1 > 0:
        P_R = P_R_1
        P_theta = P_theta_1
    else:
        P_R = P_R_2
        P_theta = P_theta_2
        
    return np.array([R, theta, P_R, P_theta], dtype=float)

def discretize_trajectory_to_grid(trajectory, grid_params):
    """
    Project 4D trajectory [R, theta, P_R, P_theta] onto PSOS grid (theta, P_psi).
    """
    nx = int(grid_params['grid_resolution_x'])
    ny = int(grid_params['grid_resolution_y'])
    psi_min, psi_max = grid_params['x_min'], grid_params['x_max']
    ppsi_min, ppsi_max = grid_params['y_min'], grid_params['y_max']
    
    R = trajectory[:, 0]
    theta = trajectory[:, 1]
    P_R = trajectory[:, 2]
    P_theta = trajectory[:, 3]
    
    # Calculate MEP R_e(theta) and derivatives to get P_psi
    # We need to do this vector-wise for efficiency
    
    dR_eq_dtheta = (-0.2551*np.sin(theta) - 2.0*0.49830*np.sin(2*theta) + 3.0*0.0534270*np.sin(3*theta) 
                    + 4.0*0.0681241*np.sin(4*theta) - 5.0*0.0205782*np.sin(5*theta))
    
    # P_psi = P_theta - R_e'(theta) * P_R
    P_psi = P_theta - (dR_eq_dtheta * P_R)
    
    cols = ((theta - psi_min) / (psi_max - psi_min) * nx).astype(int)
    rows = ((P_psi - ppsi_min) / (ppsi_max - ppsi_min) * ny).astype(int)
    
    valid_mask = (cols >= 0) & (cols < nx) & (rows >= 0) & (rows < ny)
    
    return list(zip(rows[valid_mask], cols[valid_mask]))
