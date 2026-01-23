"""
NumPy ADAPT-style gate program runner (with jac=True + ADAPT gradient-based operator selection)

- No classes; pass arrays directly.
- Supports RX / RY / RZ and RZZ
- Qubit indexing: q = 0 is least-significant bit (LSB)

Stopping rule (MODIFIED):
- Stop when fidelity changes less than eps_fid for `patience` consecutive ADAPT steps.
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# -----------------------------
# Op-codes
# -----------------------------
OP_RX  = 0
OP_RY  = 1
OP_RZ  = 2
OP_RZZ = 3

# Parameter-shift amount
SHIFT = np.pi / 2


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
# jac=True via parameter-shift (FULL gradient for SciPy)
# -----------------------------
def value_and_grad(p, n_qubits, op_codes, q1, q2, target_state, signs):
    """
    Return (loss(p), grad_loss(p)) for SciPy with jac=True.
    Parameter-shift:
      dL/dp_k = 0.5*(L(p+SHIFT e_k) - L(p-SHIFT e_k))
    """
    p = np.asarray(p, dtype=float)

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


def optimize_params(
    n_qubits,
    params,
    op_codes,
    q1,
    q2,
    target_state,
    method="L-BFGS-B",
    maxiter=200,
    signs=None,
    disp=False,
):
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
# ADAPT: gradient-based operator selection
# -----------------------------
def find_best_op(
    n_qubits,
    params,
    op_codes,
    q1,
    q2,
    target_state,
    pool,
    method="L-BFGS-B",
    maxiter=200,
    signs=None,
    grad_eps=1e-6,
):
    if signs is None:
        signs = precompute_rzz_signs(n_qubits)

    # 1) Optimize current circuit once
    if len(params) > 0:
        opt_old = optimize_params(
            n_qubits, params, op_codes, q1, q2, target_state,
            method=method, maxiter=maxiter, signs=signs, disp=False
        )
        params_opt = np.array(opt_old.x, dtype=float)
    else:
        params_opt = np.array(params, dtype=float)

    # 2) Score candidates by gradient of NEW parameter
    best_score = 0.0
    best_choice = None

    for (op, qq1, qq2) in pool:
        op_codes_ext = np.append(op_codes, op)
        q1_ext = np.append(q1, qq1)
        q2_ext = np.append(q2, qq2)

        p0 = np.append(params_opt, 0.0)

        p_plus = p0.copy()
        p_minus = p0.copy()
        p_plus[-1] += SHIFT
        p_minus[-1] -= SHIFT

        lp = loss(n_qubits, p_plus, op_codes_ext, q1_ext, q2_ext, target_state, signs=signs)
        lm = loss(n_qubits, p_minus, op_codes_ext, q1_ext, q2_ext, target_state, signs=signs)

        grad_new = 0.5 * (lp - lm)
        score = float(np.abs(grad_new))

        if score > best_score:
            best_score = score
            best_choice = (op, qq1, qq2)

    if best_choice is None or best_score < grad_eps:
        return params_opt, op_codes, q1, q2, 0.0

    # 3) Append best gate and optimize ONCE
    op_best, qq1_best, qq2_best = best_choice

    op_codes_new = np.append(op_codes, op_best)
    q1_new = np.append(q1, qq1_best)
    q2_new = np.append(q2, qq2_best)
    params_new0 = np.append(params_opt, 0.0)

    opt_new = optimize_params(
        n_qubits, params_new0, op_codes_new, q1_new, q2_new, target_state,
        method=method, maxiter=maxiter, signs=signs, disp=False
    )
    params_new = np.array(opt_new.x, dtype=float)

    return params_new, op_codes_new, q1_new, q2_new, best_score


def adapt_vqe(
    n_qubits,
    target,
    pool,
    max_op,
    eps_fid=1e-4,     # <-- fidelity threshold
    patience=3,       # <-- number of consecutive small-change steps to stop
    method="L-BFGS-B",
    maxiter=200,
    grad_eps=1e-6,
):
    """
    Stop when |F_i - F_{i-1}| < eps_fid for `patience` consecutive steps.
    """
    signs = precompute_rzz_signs(n_qubits)

    params   = np.array([], dtype=float)
    op_codes = np.array([], dtype=int)
    q1       = np.array([], dtype=int)
    q2       = np.array([], dtype=int)

    prev_fid = 0.0
    small_change_count = 0

    for i in range(max_op):
        # ADAPT step
        params, op_codes, q1, q2, score = find_best_op(
            n_qubits,
            params,
            op_codes,
            q1,
            q2,
            target,
            pool=pool,
            method=method,
            maxiter=maxiter,
            signs=signs,
            grad_eps=grad_eps,
        )

        # Compute fidelity after this step (loss = -fidelity)
        fid = -loss(n_qubits, params, op_codes, q1, q2, target, signs=signs)
        delta = abs(fid - prev_fid)

        print(
            f"step {i+1:2d}/{max_op} | progress: {100*(i+1)/max_op:6.1f}%"
            f" | score: {score:.3e} | fidelity: {fid:.8f} | Δfid: {delta:.3e}"
            f" | smallΔ: {small_change_count}/{patience}"
        )

        # Update consecutive-small-change counter
        if i > 0 and delta < eps_fid:
            small_change_count += 1
        else:
            small_change_count = 0

        # Stop if small change persisted for patience steps
        if small_change_count >= patience:
            break

        prev_fid = fid

    return params, op_codes, q1, q2


# -----------------------------
# Main (complete): run ADAPT-VQE and plot Fidelity + |ΔF| in one figure
# -----------------------------
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    n_qubits = 12
    max_op   = 150

    # Stop when fidelity changes less than eps_fid for `patience` consecutive steps
    eps_fid  = 1e-6
    patience = 10

    # Build problem
    pool = make_default_pool(n_qubits)
    target = random_haar_state(n_qubits)
    signs = precompute_rzz_signs(n_qubits)

    # Track fidelity and its change per step
    fidelities = []
    delta_fidelities = []

    # Initialize empty ansatz
    params   = np.array([], dtype=float)
    op_codes = np.array([], dtype=int)
    q1       = np.array([], dtype=int)
    q2       = np.array([], dtype=int)

    prev_fid = 0.0
    small_count = 0

    # ADAPT loop (one operator per step)
    for i in range(max_op):
        params, op_codes, q1, q2, score = find_best_op(
            n_qubits,
            params,
            op_codes,
            q1,
            q2,
            target,
            pool=pool,
            method="L-BFGS-B",
            maxiter=200,
            signs=signs,
            grad_eps=1e-6,
        )

        fid = -loss(n_qubits, params, op_codes, q1, q2, target, signs=signs)
        fidelities.append(fid)

        delta = abs(fid - prev_fid)
        delta_fidelities.append(delta)

        print(f"step {i+1:3d}/{max_op} | fidelity: {fid:.8f} | Δfid: {delta:.2e} | score: {score:.2e}")

        # Convergence check: consecutive small changes in fidelity
        if i > 0 and delta < eps_fid:
            small_count += 1
        else:
            small_count = 0

        if small_count >= patience:
            print(f"Stopping early: |Δfid| < {eps_fid} for {patience} consecutive steps.")
            break

        prev_fid = fid

    # Final summary
    print("\nFinal results")
    print("Number of operators:", len(op_codes))
    print("Best fidelity:", fidelities[-1] if fidelities else 0.0)
    print("Best op_codes:", op_codes)
    print("Best q1:", q1)
    print("Best q2:", q2)

    # Plot: fidelity (linear) + |ΔF| (log) in the same figure
    steps = np.arange(1, len(fidelities) + 1)

    fig, ax1 = plt.subplots()

    # Fidelity on left axis (linear)
    ax1.plot(steps, fidelities, marker="o", label="Fidelity")
    ax1.set_xlabel("Number of operators")
    ax1.set_ylabel("Fidelity")
    #ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    # |ΔF| on right axis (log)
    ax2 = ax1.twinx()
    ax2.plot(steps, delta_fidelities, marker="s", linestyle="--", label="|Δ Fidelity|")
    ax2.set_ylabel("|Δ Fidelity|")
    ax2.set_yscale("log")
    #ax2.set_ylim(bottom=1e-16)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title("ADAPT-VQE convergence: fidelity and per-step improvement")
    plt.tight_layout()
    plt.show()














# if __name__ == "__main__":
#     n_qubits = 12
#     max_op   = 100

#     # Stop when fidelity changes less than eps_fid for `patience` consecutive steps
#     eps_fid  = 1e-6
#     patience = 10

#     pool = make_default_pool(n_qubits)
#     target = random_haar_state(n_qubits)

#     params, op_codes, q1, q2 = adapt_vqe(
#         n_qubits,
#         target,
#         pool,
#         max_op,
#         eps_fid=eps_fid,
#         patience=patience,
#         method="L-BFGS-B",
#         maxiter=200,
#         grad_eps=1e-6,
#     )

#     signs = precompute_rzz_signs(n_qubits)
#     best_fid = -loss(n_qubits, params, op_codes, q1, q2, target, signs=signs)

#     print("\nop_codes:", op_codes)
#     print("Best fidelity:", best_fid)
#     print("Best op_codes:", op_codes)
#     print("Best q1:", q1)
#     print("Best q2:", q2)
