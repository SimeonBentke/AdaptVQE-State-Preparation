"""
NumPy ADAPT-style gate program runner (with jac=True via parameter-shift gradient)
- No classes; pass arrays directly.
- Supports RX / RY / RZ and RZZ
- Qubit indexing: q = 0 is least-significant bit (LSB)

IMPORTANT:
- This version uses SciPy minimize(..., jac=True) by providing a function that returns (value, grad).
- The gradient is computed with the parameter-shift rule (2 circuit evaluations per parameter).
  This can be substantially more expensive per optimizer step than finite-differences for large parameter counts,
  but it is a correct "jac=True" implementation and often numerically more stable.

Also includes the two big simulator speedups:
1) 1-qubit gates applied in-place (no full statevector copy per gate).
2) RZZ sign patterns precomputed once per (q1,q2), reused every call.

Self-contained and runnable.
"""

import numpy as np
from scipy.optimize import minimize


# -----------------------------
# Op-codes
# -----------------------------
OP_RX  = 0
OP_RY  = 1
OP_RZ  = 2
OP_RZZ = 3


def make_default_pool(n_qubits):
    pool = []

    # Single-qubit rotations
    for q in range(n_qubits):
        pool.append((OP_RX, q, 0))
        pool.append((OP_RY, q, 0))
        pool.append((OP_RZ, q, 0))

    # Two-qubit ZZ rotations: all unordered pairs
    for q1 in range(n_qubits):
        for q2 in range(q1 + 1, n_qubits):
            pool.append((OP_RZZ, q1, q2))

    return pool


def random_haar_state(n_qubits, dtype=np.complex128, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    dim = 2 ** n_qubits
    psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    psi /= np.linalg.norm(psi)
    return psi.astype(dtype)


# -----------------------------
# Gate matrices (2x2)
# -----------------------------
def RX(theta):
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -1j * s],
                     [-1j * s, c]], dtype=np.complex128)


def RY(theta):
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s],
                     [s,  c]], dtype=np.complex128)


def RZ(theta):
    return np.array([[np.exp(-1j * theta / 2), 0.0],
                     [0.0, np.exp(1j * theta / 2)]], dtype=np.complex128)


# -----------------------------
# Initial state |0...0>
# -----------------------------
def init_state(n_qubits: int):
    dim = 2 ** n_qubits
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0 + 0.0j
    return psi


# -----------------------------
# Apply 1-qubit gate IN-PLACE
# -----------------------------
def apply_1q_gate_inplace(psi, U, q, n_qubits):
    """
    Apply a 2x2 gate U to qubit q (0 = LSB) IN PLACE.
    Complexity: O(2^n)
    """
    dim = psi.shape[0]
    half = dim // 2

    k = np.arange(half, dtype=np.int64)

    low_mask = (1 << q) - 1
    low = k & low_mask
    high = k >> q

    base = (high << (q + 1)) | low
    i0 = base
    i1 = base | (1 << q)

    # Copy only the touched amplitudes (size 2^(n-1)), NOT the full state
    a0 = psi[i0].copy()
    a1 = psi[i1].copy()

    psi[i0] = U[0, 0] * a0 + U[0, 1] * a1
    psi[i1] = U[1, 0] * a0 + U[1, 1] * a1


# -----------------------------
# RZZ acceleration: precompute signs once
# -----------------------------
def precompute_rzz_signs(n_qubits):
    """
    For each unordered pair (a,b), precompute s[idx] in {+1,-1} such that:
      RZZ(theta) multiplies amplitude idx by exp(-i theta/2 * s[idx]).
    """
    dim = 2 ** n_qubits
    idx = np.arange(dim, dtype=np.int64)

    signs = {}
    for a in range(n_qubits):
        for b in range(a + 1, n_qubits):
            b1 = (idx >> a) & 1
            b2 = (idx >> b) & 1
            s = np.where(b1 == b2, 1.0, -1.0)
            signs[(a, b)] = s
    return signs


def rzz_apply_inplace(psi, theta, q1, q2, signs):
    """
    Apply RZZ(theta) IN PLACE using precomputed signs.
    """
    a, b = (q1, q2) if q1 < q2 else (q2, q1)
    s = signs[(a, b)]
    psi *= np.exp(-1j * theta / 2 * s)


# -----------------------------
# Circuit execution
# -----------------------------
def apply_unitary(n_qubits, params, op_codes, q1, q2, signs=None):
    psi = init_state(n_qubits)

    for i in range(len(params)):
        theta = float(params[i])
        op    = int(op_codes[i])

        if op == OP_RX:
            apply_1q_gate_inplace(psi, RX(theta), int(q1[i]), n_qubits)

        elif op == OP_RY:
            apply_1q_gate_inplace(psi, RY(theta), int(q1[i]), n_qubits)

        elif op == OP_RZ:
            apply_1q_gate_inplace(psi, RZ(theta), int(q1[i]), n_qubits)

        elif op == OP_RZZ:
            if signs is None:
                # Correct fallback (slower)
                dim = psi.shape[0]
                idx = np.arange(dim, dtype=np.int64)
                b1 = (idx >> int(q1[i])) & 1
                b2 = (idx >> int(q2[i])) & 1
                s = np.where(b1 == b2, 1.0, -1.0)
                psi *= np.exp(-1j * theta / 2 * s)
            else:
                rzz_apply_inplace(psi, theta, int(q1[i]), int(q2[i]), signs)
        else:
            raise ValueError(f"Unknown op-code {op}")

    return psi


# -----------------------------
# Objective: loss = - fidelity
# -----------------------------
def fidelity(n_qubits, params, op_codes, q1, q2, target_state, signs=None):
    psi = apply_unitary(n_qubits, params, op_codes, q1, q2, signs=signs)
    return abs(np.vdot(target_state, psi)) ** 2


def loss(n_qubits, params, op_codes, q1, q2, target_state, signs=None):
    return -fidelity(n_qubits, params, op_codes, q1, q2, target_state, signs=signs)


# -----------------------------
# jac=True via parameter-shift
# -----------------------------
SHIFT = np.pi / 2


def value_and_grad(p, n_qubits, op_codes, q1, q2, target_state, signs):
    """
    Return (loss(p), grad_loss(p)) for SciPy with jac=True.
    Parameter-shift: grad_k = 0.5*(L(p+shift e_k) - L(p-shift e_k)).
    """
    p = np.asarray(p, dtype=float)

    # base value
    base = loss(n_qubits, p, op_codes, q1, q2, target_state, signs=signs)

    grad = np.zeros_like(p)
    for k in range(p.size):
        p_plus = p.copy()
        p_minus = p.copy()
        p_plus[k] += SHIFT
        p_minus[k] -= SHIFT

        lp = loss(n_qubits, p_plus, op_codes, q1, q2, target_state, signs=signs)
        lm = loss(n_qubits, p_minus, op_codes, q1, q2, target_state, signs=signs)

        grad[k] = 0.5 * (lp - lm)

    return base, grad


def optimize_params(n_qubits, params, op_codes, q1, q2, target_state,
                    method="L-BFGS-B", maxiter=200, signs=None, disp=False):
    """
    Minimize loss(params) using SciPy with jac=True.
    """
    if signs is None:
        signs = precompute_rzz_signs(n_qubits)

    x0 = np.array(params, dtype=float)

    result = minimize(
        fun=value_and_grad,
        x0=x0,
        args=(n_qubits, op_codes, q1, q2, target_state, signs),
        method=method,
        jac=True,
        options={"maxiter": maxiter, "disp": disp},
    )
    return result


# -----------------------------
# ADAPT outer loop (naive selection: optimize each pool element)
# -----------------------------
def find_best_op(n_qubits, params, op_codes, q1, q2, target_state, pool,
                 method="L-BFGS-B", maxiter=200, signs=None):

    if signs is None:
        signs = precompute_rzz_signs(n_qubits)

    opt_old = optimize_params(n_qubits, params, op_codes, q1, q2, target_state,
                              method=method, maxiter=maxiter, signs=signs, disp=False)
    old_loss = float(opt_old.fun)

    best_improvement = 0.0
    best_choice = None
    best_params_full = None

    for (op, qq1, qq2) in pool:
        params_new   = np.append(params, 0.0)
        op_codes_new = np.append(op_codes, op)
        q1_new       = np.append(q1, qq1)
        q2_new       = np.append(q2, qq2)

        opt_new = optimize_params(n_qubits, params_new, op_codes_new, q1_new, q2_new, target_state,
                                  method=method, maxiter=maxiter, signs=signs, disp=False)
        new_loss = float(opt_new.fun)

        improvement = old_loss - new_loss

        if improvement > best_improvement:
            best_improvement = improvement
            best_choice = (op_codes_new, q1_new, q2_new)
            best_params_full = opt_new.x

    if best_choice is None:
        return params, op_codes, q1, q2, 0.0

    op_codes_new, q1_new, q2_new = best_choice
    params_new = best_params_full
    return params_new, op_codes_new, q1_new, q2_new, best_improvement


def adapt_vqe(n_qubits, target, pool, max_op, eps=0.01, method="L-BFGS-B", maxiter=200):
    signs = precompute_rzz_signs(n_qubits)

    params   = np.array([], dtype=float)
    op_codes = np.array([], dtype=int)
    q1       = np.array([], dtype=int)
    q2       = np.array([], dtype=int)

    for i in range(max_op):
        print(f"operation: {i} ----- {i/max_op*100:.1f} %\n")

        params, op_codes, q1, q2, imp = find_best_op(
            n_qubits, params, op_codes, q1, q2, target, pool=pool,
            method=method, maxiter=maxiter, signs=signs
        )

        if imp < eps:
            return params, op_codes, q1, q2

    return params, op_codes, q1, q2


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    n_qubits = 12
    max_op   = 15
    eps      = 0.0

    pool = make_default_pool(n_qubits)
    target = random_haar_state(n_qubits)

    params, op_codes, q1, q2 = adapt_vqe(
        n_qubits, target, pool, max_op, eps=eps,
        method="L-BFGS-B", maxiter=200
    )

    signs = precompute_rzz_signs(n_qubits)
    best_loss = loss(n_qubits, params, op_codes, q1, q2, target, signs=signs)

    print("\nop_codes:", op_codes)
    print("Best fidelity:", -best_loss)
    print("Best op_codes:", op_codes)
    print("Best q1:", q1)
    print("Best q2:", q2)
