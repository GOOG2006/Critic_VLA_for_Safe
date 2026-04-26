"""CBF (Control Barrier Function) utilities for ellipsoid-ellipsoid safety.

Vendored from vlsa-aegis/main/utils.py (AEGIS project).
Pure numpy + cvxpy, no external simulation dependencies.
"""
import os
import warnings
import numpy as np
import cvxpy as cp

# Emit a warning whenever the QP falls back to the reference action.
# Set CBF_QP_WARN=0 to silence (useful for large sweeps).
_CBF_QP_WARN = os.environ.get("CBF_QP_WARN", "1") not in ("0", "false", "False")


def vector_hat(v):
    """Skew-symmetric matrix from 3D vector."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)


def project_matrix(z):
    """Projection matrix onto plane perpendicular to z."""
    z = np.asarray(z, dtype=np.float64).reshape(3, 1)
    return np.eye(3) - z @ z.T


def compute_h_ij(p_i, Q_i_diag, R_i, p_j, Q_j_diag, R_j, z_ij):
    """Compute CBF h value between two ellipsoids.

    Args:
        p_i: center of ellipsoid i (gripper), shape (3,)
        Q_i_diag: semi-axis lengths of ellipsoid i, shape (3,)
        R_i: rotation matrix of ellipsoid i, shape (3,3)
        p_j: center of ellipsoid j (obstacle), shape (3,)
        Q_j_diag: semi-axis lengths of ellipsoid j, shape (3,)
        R_j: rotation matrix of ellipsoid j, shape (3,3)
        z_ij: direction vector (typically p_j - p_i), shape (3,)

    Returns:
        h: scalar barrier value. h > 0 means safe, h < 0 means collision.
    """
    Q_i = np.diag(Q_i_diag); Q_j = np.diag(Q_j_diag)
    Qbar_i = R_i @ Q_i @ R_i.T
    Qbar_j = R_j @ Q_j @ R_j.T
    Qbar_i_inv = np.linalg.inv(Qbar_i)
    z = z_ij / max(np.linalg.norm(z_ij), 1e-10)
    term1 = np.linalg.norm(Qbar_j @ Qbar_i_inv @ z)
    term2 = (p_j - p_i).T @ Qbar_i_inv @ z
    denom = np.linalg.norm(Qbar_i_inv @ z) + 1e-10
    return float((-term1 + term2 - 1.0) / denom)


def compute_h_coeffs_3d(p_i, Q_i_diag, R_i, p_j, Q_j_diag, R_j, z, eps=1e-10):
    """Compute CBF value and gradient coefficients for 9-DoF QP.

    Returns:
        a_v: velocity constraint coefficients (3,) in local frame
        a_omega: angular velocity constraint coefficients (3,) in local frame
        a_uz: z-direction steering coefficients (3,)
        h: barrier value (scalar)
        mu_row: dh/dz row vector (3,)
    """
    Q_i = np.diag(Q_i_diag); Q_j = np.diag(Q_j_diag)
    Qbar_i = R_i @ Q_i @ R_i.T
    Qbar_j = R_j @ Q_j @ R_j.T
    Qbar_i_inv = np.linalg.inv(Qbar_i)
    Qbar_i_inv2 = Qbar_i_inv @ Qbar_i_inv
    Qbar_j2 = Qbar_j @ Qbar_j
    z = z / (np.linalg.norm(z) + eps)
    a_vec = Qbar_i_inv @ z
    denom = np.linalg.norm(a_vec) + eps
    b_vec = Qbar_j @ a_vec
    term1 = np.linalg.norm(b_vec) + eps
    sigma = term1 * denom + eps
    rho = (1.0 - (p_j - p_i).T @ a_vec + term1)
    eta_row = -(1.0 / denom) * (z.T @ Qbar_i_inv)
    term_mu_1 = (rho / (denom**3 + eps)) * (z.T @ Qbar_i_inv2)
    term_mu_2 = (1.0 / denom) * ((p_j - p_i).T @ Qbar_i_inv)
    term_mu_3 = (1.0 / sigma) * (z.T @ Qbar_i_inv @ Qbar_j2 @ Qbar_i_inv)
    mu_row = term_mu_1 + term_mu_2 - term_mu_3
    tmp1 = z.T @ Qbar_i_inv2 @ vector_hat(z)
    left_vec = (z.T @ Qbar_i_inv @ Qbar_j2)
    Ja_vec = vector_hat(a_vec)
    tmp2 = left_vec @ (Ja_vec - Qbar_i_inv @ vector_hat(z))
    part_a = (p_j - p_i).T @ Qbar_i_inv @ vector_hat(z)
    part_b = z.T @ Qbar_i_inv @ vector_hat(p_j - p_i)
    tmp3 = part_a + part_b
    zeta_tilde = rho * (1.0 / (denom**3 + eps)) * tmp1 + (1.0 / sigma) * tmp2 + (1.0 / denom) * tmp3
    a_v = (eta_row @ R_i).ravel()
    a_omega = zeta_tilde @ R_i
    a_uz = (mu_row @ project_matrix(z)).ravel()
    h = compute_h_ij(p_i, Q_i_diag, R_i, p_j, Q_j_diag, R_j, z)
    return a_v, a_omega, a_uz, h, mu_row


def solve_cbf_qp(v_ref, omega_ref, uz_ref, p1, Q1, R1, p2, Q2, R2, z_fixed, alpha=10.0):
    """Solve AEGIS-style 9-DoF QP with CBF constraint.

    Args:
        v_ref: reference velocity in EE local frame (3,)
        omega_ref: reference angular velocity (3,)
        uz_ref: reference z-direction steering (3,)
        p1, Q1, R1: gripper ellipsoid
        p2, Q2, R2: obstacle ellipsoid
        z_fixed: current escape direction (3,)
        alpha: CBF gain (default 10.0, AEGIS default)

    Returns:
        u_v, u_omega, u_z: optimized variables
        h: barrier value
        status: one of "ok", "infeasible", "solver_error:<msg>". When status != "ok",
                the returned u_v/u_omega/u_z are the *reference* action — i.e. the CBF
                filter did NOT actually enforce the barrier constraint, and the caller
                must record this.
    """
    a_v, a_omega, a_uz, h, mu_row = compute_h_coeffs_3d(p1, Q1, R1, p2, Q2, R2, z_fixed)

    u = cp.Variable(9)
    W = np.diag([1./25]*6 + [1.]*3)
    u_ref = np.hstack([v_ref * 5.0, omega_ref * 5.0, uz_ref * 10.0])
    cost = cp.quad_form(u - u_ref, W)
    a_u_v = 0.2 * a_v
    a_u_omega = 0.2 * a_omega
    constraints = [a_u_v @ u[:3] + a_u_omega @ u[3:6] + a_uz @ u[6:] + alpha * h >= 0]

    prob = cp.Problem(cp.Minimize(cost), constraints)
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
    except Exception as e:
        if _CBF_QP_WARN:
            warnings.warn(f"[cbf_qp] solver error h={h:.4f}: {e}; falling back to u_ref", stacklevel=2)
        return v_ref, omega_ref, uz_ref, h, f"solver_error:{type(e).__name__}"

    if u.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        if _CBF_QP_WARN:
            warnings.warn(
                f"[cbf_qp] infeasible/no-solution (status={prob.status}, h={h:.4f}); "
                f"falling back to u_ref — barrier NOT enforced",
                stacklevel=2,
            )
        return v_ref, omega_ref, uz_ref, h, "infeasible"

    return u.value[:3], u.value[3:6], u.value[6:], h, "ok"
