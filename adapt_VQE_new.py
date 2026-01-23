"""
NumPy ADAPT-style gate program runner
- No classes; pass arrays directly.
- Supports RX / RY / RZ and RZZ
- Qubit indexing: q = 0 is least-significant bit (LSB)

This file is self-contained and runnable.
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
    """
    Generate a Haar-random n-qubit pure state.

    Returns
    -------
    psi : ndarray, shape (2**n_qubits,)
        Normalized complex statevector.
    """
    if rng is None:
        rng = np.random.default_rng()

    dim = 2 ** n_qubits

    real = rng.normal(size=dim)
    imag = rng.normal(size=dim)

    psi = real + 1j * imag
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
# Apply 1-qubit gate efficiently
# -----------------------------
def apply_1q_gate(psi, U, q, n_qubits):
    """
    Apply a 2x2 gate U to qubit q (0 = LSB).
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

    a0 = psi[i0]
    a1 = psi[i1]

    b0 = U[0, 0] * a0 + U[0, 1] * a1
    b1 = U[1, 0] * a0 + U[1, 1] * a1

    psi2 = psi.copy()
    psi2[i0] = b0
    psi2[i1] = b1
    return psi2


# -----------------------------
# Apply RZZ gate
# -----------------------------
def apply_rzz(psi, theta, q1, q2, n_qubits):
    """
    Apply RZZ(theta) = exp(-i theta/2 * Z⊗Z)
    """
    dim = psi.shape[0]
    idx = np.arange(dim, dtype=np.int64)

    b1 = (idx >> q1) & 1
    b2 = (idx >> q2) & 1

    s = np.where(b1 == b2, 1.0, -1.0)
    phase = np.exp(-1j * theta / 2 * s)

    return psi * phase


# -----------------------------
# Main runner
# -----------------------------
def apply_unitary(n_qubits,params,op_codes,q1,q2):
    """
    Execute a gate program.

    params   : (L,)
    op_codes : (L,)
    q1, q2   : (L,)
    """
    psi = init_state(n_qubits)

    for i in range(len(params)):
        theta = params[i]
        op    = op_codes[i]

        if op == OP_RX:
            psi = apply_1q_gate(psi, RX(theta), q1[i], n_qubits)

        elif op == OP_RY:
            psi = apply_1q_gate(psi, RY(theta), q1[i], n_qubits)

        elif op == OP_RZ:
            psi = apply_1q_gate(psi, RZ(theta), q1[i], n_qubits)

        elif op == OP_RZZ:
            psi = apply_rzz(psi, theta, q1[i], q2[i], n_qubits)

        else:
            raise ValueError(f"Unknown op-code {op}")

    return psi

def loss(n_qubits,params,op_codes,q1,q2,target_state):
    psi = apply_unitary(n_qubits=n_qubits,params=params,op_codes=op_codes,q1=q1,q2=q2)
    return -abs(np.vdot(target_state, psi)) ** 2



def optimize_params(n_qubits, params, op_codes, q1, q2, target_state,
                    method="L-BFGS-B", maxiter=200):
    """
    Minimize loss(params) using SciPy.

    Returns: OptimizeResult
    """
    def f(p):
        return loss(n_qubits=n_qubits, params=p, op_codes=op_codes, q1=q1, q2=q2, target_state=target_state)

    result = minimize(
        f,
        x0=np.array(params, dtype=float),
        method=method,
        options={"maxiter": maxiter, "disp": True},
    )
    return result




def find_best_op(n_qubits, params, op_codes, q1, q2, target_state, pool,
                 method="L-BFGS-B", maxiter=200):

    opt_old = optimize_params(n_qubits, params, op_codes, q1, q2, target_state,
                              method=method, maxiter=maxiter)
    old_loss = opt_old.fun

    best_improvement = 0.0
    best_choice = None
    best_params_full = None

    for (op, qq1, qq2) in pool:
        params_new   = np.append(params, 0.0)
        op_codes_new = np.append(op_codes, op)
        q1_new       = np.append(q1, qq1)
        q2_new       = np.append(q2, qq2)

        opt_new = optimize_params(n_qubits, params_new, op_codes_new, q1_new, q2_new, target_state,
                                  method=method, maxiter=maxiter)
        new_loss = opt_new.fun

        improvement = old_loss - new_loss  # positive means better (loss decreased)

        if improvement > best_improvement:
            best_improvement = improvement
            best_choice = (op, qq1, qq2, op_codes_new, q1_new, q2_new)
            best_params_full = opt_new.x     # FULL optimized param vector

    if best_choice is None:
        # no improvement found; return original
        return params, op_codes, q1, q2

    op, qq1, qq2, op_codes_new, q1_new, q2_new = best_choice

    # IMPORTANT: replace params with the full optimized vector (do NOT append)
    params_new = best_params_full

    return params_new, op_codes_new, q1_new, q2_new, best_improvement

def adapt_vqe(n_qubits, target, pool, max_op, eps=0.01, method="L-BFGS-B", maxiter=200):
    params  = np.array([], dtype=float)
    op_codes = np.array([], dtype=int)
    q1       = np.array([], dtype=int)
    q2       = np.array([], dtype=int)

    for i in range(max_op):
        print("operation: ", i,"-----", i/max_op*100, " %")
        print()
        params, op_codes, q1, q2, imp=find_best_op(n_qubits, params, op_codes, q1, q2, target, pool=pool, 
                                              method="L-BFGS-B", maxiter=200)
        if imp<eps:
            return params, op_codes, q1, q2
    return params, op_codes, q1, q2
        






if __name__ == "__main__":
    n_qubits = 12
    max_op=20
    eps=0
    pool=make_default_pool(n_qubits)

    target = random_haar_state(n_qubits)

    params, op_codes, q1, q2=adapt_vqe(n_qubits, target, pool, max_op, eps=eps, 
                                       method="L-BFGS-B", maxiter=200)

    
    best_loss=loss(n_qubits,params,op_codes,q1,q2,target)
    


    
    print("\nop_codes:", op_codes)
    print("Best fidelity:", -best_loss)
    #print("Best params:", params)
    print("Best op_codes:", op_codes)
    print("Best q1:", q1)
    print("Best q2:", q2)
