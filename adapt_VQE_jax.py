import os
os.environ.setdefault("JAX_ENABLE_X64", "True")

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, vmap
from functools import partial

from qiskit import QuantumCircuit


# ============================================================
# Op-codes
# ============================================================
OP_RX  = 0
OP_RY  = 1
OP_RZ  = 2
OP_RZZ = 3


# ============================================================
# Pool construction
# ============================================================
def make_default_pool(n_qubits):
    pool = []

    # Single-qubit rotations
    for q in range(n_qubits):
        pool.append((OP_RX, q, 0))
        pool.append((OP_RY, q, 0))
        pool.append((OP_RZ, q, 0))

    # Two-qubit ZZ rotations
    for qa in range(n_qubits):
        for qb in range(qa + 1, n_qubits):
            pool.append((OP_RZZ, qa, qb))

    return pool


def pool_to_arrays(pool):
    ops = np.array([p[0] for p in pool], dtype=np.int32)
    q1  = np.array([p[1] for p in pool], dtype=np.int32)
    q2  = np.array([p[2] for p in pool], dtype=np.int32)
    return jnp.asarray(ops), jnp.asarray(q1), jnp.asarray(q2)


# ============================================================
# States
# ============================================================
def init_state(n_qubits):
    dim = 1 << n_qubits
    psi = jnp.zeros((dim,), dtype=jnp.complex128)
    psi = psi.at[0].set(1.0 + 0.0j)
    return psi


def random_haar_state(n_qubits, key):
    dim = 1 << n_qubits
    key_r, key_i = jax.random.split(key)
    x = jax.random.normal(key_r, (dim,), dtype=jnp.float64)
    y = jax.random.normal(key_i, (dim,), dtype=jnp.float64)
    psi = x + 1j * y
    psi = psi / jnp.linalg.norm(psi)
    return psi.astype(jnp.complex128)


# ============================================================
# Gate matrices
# ============================================================
def RX(theta):
    c = jnp.cos(theta / 2.0)
    s = jnp.sin(theta / 2.0)
    return jnp.array([[c, -1j * s],
                      [-1j * s, c]], dtype=jnp.complex128)


def RY(theta):
    c = jnp.cos(theta / 2.0)
    s = jnp.sin(theta / 2.0)
    return jnp.array([[c, -s],
                      [s,  c]], dtype=jnp.complex128)


def RZ(theta):
    return jnp.array([[jnp.exp(-1j * theta / 2.0), 0.0],
                      [0.0, jnp.exp(1j * theta / 2.0)]], dtype=jnp.complex128)


# ============================================================
# Precomputation
# ============================================================
def precompute_1q_indices(n_qubits):
    """
    For each qubit q, precompute index arrays i0[q], i1[q]
    such that amplitudes at i0/i1 form the pairs touched by
    a 1-qubit gate on qubit q (q = 0 is LSB).
    """
    dim = 1 << n_qubits
    half = dim >> 1
    k = np.arange(half, dtype=np.int64)

    i0_list = []
    i1_list = []

    for q in range(n_qubits):
        low_mask = (1 << q) - 1
        low = k & low_mask
        high = k >> q
        base = (high << (q + 1)) | low
        i0 = base
        i1 = base | (1 << q)
        i0_list.append(i0)
        i1_list.append(i1)

    i0 = jnp.asarray(np.stack(i0_list), dtype=jnp.int32)
    i1 = jnp.asarray(np.stack(i1_list), dtype=jnp.int32)
    return i0, i1


def build_sign_tensor(n_qubits):
    """
    sign_tensor[a, b, idx] in {+1, -1} for a < b,
    such that RZZ(theta) on (a,b) multiplies amplitude idx by
      exp(-i theta/2 * sign_tensor[a,b,idx])
    """
    dim = 1 << n_qubits
    idx = np.arange(dim, dtype=np.int64)

    signs = np.zeros((n_qubits, n_qubits, dim), dtype=np.float64)

    for a in range(n_qubits):
        for b in range(a + 1, n_qubits):
            b1 = (idx >> a) & 1
            b2 = (idx >> b) & 1
            s = np.where(b1 == b2, 1.0, -1.0)
            signs[a, b, :] = s

    return jnp.asarray(signs, dtype=jnp.float64)


# ============================================================
# Gate application
# ============================================================
def apply_1q_gate(psi, U, q, i0_table, i1_table):
    i0 = i0_table[q]
    i1 = i1_table[q]

    a0 = psi[i0]
    a1 = psi[i1]

    out = psi
    out = out.at[i0].set(U[0, 0] * a0 + U[0, 1] * a1)
    out = out.at[i1].set(U[1, 0] * a0 + U[1, 1] * a1)
    return out


def apply_rzz(psi, theta, q1, q2, sign_tensor):
    a = jnp.minimum(q1, q2)
    b = jnp.maximum(q1, q2)
    s = sign_tensor[a, b]
    return psi * jnp.exp(-1j * theta / 2.0 * s)


def apply_op(psi, theta, op, qa, qb, i0_table, i1_table, sign_tensor):
    def do_rx(_):
        return apply_1q_gate(psi, RX(theta), qa, i0_table, i1_table)

    def do_ry(_):
        return apply_1q_gate(psi, RY(theta), qa, i0_table, i1_table)

    def do_rz(_):
        return apply_1q_gate(psi, RZ(theta), qa, i0_table, i1_table)

    def do_rzz(_):
        return apply_rzz(psi, theta, qa, qb, sign_tensor)

    return jax.lax.switch(op, [do_rx, do_ry, do_rz, do_rzz], operand=None)


# ============================================================
# Circuit execution
# ============================================================
@partial(jit, static_argnames=("n_qubits",))
def apply_unitary(n_qubits, params, op_codes, q1, q2, i0_table, i1_table, sign_tensor):
    psi0 = init_state(n_qubits)

    def body(psi, xs):
        theta, op, qa, qb = xs
        psi = apply_op(psi, theta, op, qa, qb, i0_table, i1_table, sign_tensor)
        return psi, None

    psi_final, _ = jax.lax.scan(body, psi0, (params, op_codes, q1, q2))
    return psi_final


@partial(jit, static_argnames=("n_qubits",))
def fidelity(n_qubits, params, op_codes, q1, q2, target_state, i0_table, i1_table, sign_tensor):
    psi = apply_unitary(n_qubits, params, op_codes, q1, q2, i0_table, i1_table, sign_tensor)
    return jnp.abs(jnp.vdot(target_state, psi)) ** 2


@partial(jit, static_argnames=("n_qubits",))
def loss(n_qubits, params, op_codes, q1, q2, target_state, i0_table, i1_table, sign_tensor):
    return -fidelity(n_qubits, params, op_codes, q1, q2, target_state, i0_table, i1_table, sign_tensor)


def make_loss_fn(n_qubits, op_codes, q1, q2, target_state, i0_table, i1_table, sign_tensor):
    def f(params):
        return loss(n_qubits, params, op_codes, q1, q2, target_state, i0_table, i1_table, sign_tensor)
    return f


# ============================================================
# Optimization
# ============================================================
def optimize_params(
    n_qubits,
    params,
    op_codes,
    q1,
    q2,
    target_state,
    i0_table,
    i1_table,
    sign_tensor,
    method="L-BFGS-B",
    maxiter=200,
    disp=False,
):
    x0 = np.array(params, dtype=np.float64)

    loss_fn = make_loss_fn(
        n_qubits,
        op_codes,
        q1,
        q2,
        target_state,
        i0_table,
        i1_table,
        sign_tensor,
    )
    vg = jit(value_and_grad(loss_fn))

    def fun_np(x):
        val, _ = vg(jnp.asarray(x, dtype=jnp.float64))
        return float(val)

    def jac_np(x):
        _, grad = vg(jnp.asarray(x, dtype=jnp.float64))
        return np.array(grad, dtype=np.float64)

    result = minimize(
        fun=fun_np,
        x0=x0,
        jac=jac_np,
        method=method,
        options={"maxiter": maxiter, "disp": disp},
    )
    return result


# ============================================================
# Candidate scoring for ADAPT
# ============================================================
@partial(jit, static_argnames=("n_qubits",))
def candidate_grad_score(
    n_qubits,
    params_opt,
    op_codes,
    q1,
    q2,
    cand_op,
    cand_q1,
    cand_q2,
    target_state,
    i0_table,
    i1_table,
    sign_tensor,
):
    op_codes_ext = jnp.concatenate([op_codes, jnp.array([cand_op], dtype=op_codes.dtype)])
    q1_ext       = jnp.concatenate([q1,       jnp.array([cand_q1], dtype=q1.dtype)])
    q2_ext       = jnp.concatenate([q2,       jnp.array([cand_q2], dtype=q2.dtype)])
    p0           = jnp.concatenate([params_opt, jnp.array([0.0], dtype=params_opt.dtype)])

    def ext_loss(p):
        return loss(
            n_qubits, p, op_codes_ext, q1_ext, q2_ext,
            target_state, i0_table, i1_table, sign_tensor
        )

    g = value_and_grad(ext_loss)(p0)[1]
    return jnp.abs(g[-1])


def score_pool(
    n_qubits,
    params_opt,
    op_codes,
    q1,
    q2,
    pool_ops,
    pool_q1,
    pool_q2,
    target_state,
    i0_table,
    i1_table,
    sign_tensor,
):
    batched = vmap(
        lambda op, a, b: candidate_grad_score(
            n_qubits,
            params_opt,
            op_codes,
            q1,
            q2,
            op,
            a,
            b,
            target_state,
            i0_table,
            i1_table,
            sign_tensor,
        )
    )
    return batched(pool_ops, pool_q1, pool_q2)


def find_best_op(
    n_qubits,
    params,
    op_codes,
    q1,
    q2,
    target_state,
    pool_ops,
    pool_q1,
    pool_q2,
    i0_table,
    i1_table,
    sign_tensor,
    method="L-BFGS-B",
    maxiter=200,
    grad_eps=1e-6,
):
    if len(params) > 0:
        opt_old = optimize_params(
            n_qubits=n_qubits,
            params=params,
            op_codes=op_codes,
            q1=q1,
            q2=q2,
            target_state=target_state,
            i0_table=i0_table,
            i1_table=i1_table,
            sign_tensor=sign_tensor,
            method=method,
            maxiter=maxiter,
            disp=False,
        )
        params_opt = jnp.asarray(np.array(opt_old.x, dtype=np.float64))
    else:
        params_opt = jnp.asarray(np.array(params, dtype=np.float64))

    scores = score_pool(
        n_qubits=n_qubits,
        params_opt=params_opt,
        op_codes=op_codes,
        q1=q1,
        q2=q2,
        pool_ops=pool_ops,
        pool_q1=pool_q1,
        pool_q2=pool_q2,
        target_state=target_state,
        i0_table=i0_table,
        i1_table=i1_table,
        sign_tensor=sign_tensor,
    )

    best_idx = int(jnp.argmax(scores))
    best_score = float(scores[best_idx])

    if best_score < grad_eps:
        return (
            np.array(params_opt, dtype=np.float64),
            np.array(op_codes, dtype=np.int32),
            np.array(q1, dtype=np.int32),
            np.array(q2, dtype=np.int32),
            0.0,
        )

    op_best = int(pool_ops[best_idx])
    q1_best = int(pool_q1[best_idx])
    q2_best = int(pool_q2[best_idx])

    op_codes_new = np.append(np.array(op_codes, dtype=np.int32), op_best)
    q1_new       = np.append(np.array(q1, dtype=np.int32), q1_best)
    q2_new       = np.append(np.array(q2, dtype=np.int32), q2_best)
    params_new0  = np.append(np.array(params_opt, dtype=np.float64), 0.0)

    opt_new = optimize_params(
        n_qubits=n_qubits,
        params=params_new0,
        op_codes=jnp.asarray(op_codes_new),
        q1=jnp.asarray(q1_new),
        q2=jnp.asarray(q2_new),
        target_state=target_state,
        i0_table=i0_table,
        i1_table=i1_table,
        sign_tensor=sign_tensor,
        method=method,
        maxiter=maxiter,
        disp=False,
    )

    params_new = np.array(opt_new.x, dtype=np.float64)

    return params_new, op_codes_new, q1_new, q2_new, best_score


# ============================================================
# Circuit presentation
# ============================================================
def build_qiskit_circuit(n_qubits, params, op_codes, q1, q2):
    qc = QuantumCircuit(n_qubits)

    for theta, op, a, b in zip(params, op_codes, q1, q2):
        theta = float(theta)
        op = int(op)
        a = int(a)
        b = int(b)

        if op == OP_RX:
            qc.rx(theta, a)
        elif op == OP_RY:
            qc.ry(theta, a)
        elif op == OP_RZ:
            qc.rz(theta, a)
        elif op == OP_RZZ:
            qc.rzz(theta, a, b)
        else:
            raise ValueError(f"Unknown op-code {op}")

    return qc


def print_circuit_summary(params, op_codes, q1, q2):
    op_names = {
        OP_RX: "RX",
        OP_RY: "RY",
        OP_RZ: "RZ",
        OP_RZZ: "RZZ",
    }

    print("\nGate list:")
    print(f"{'#':>4}  {'Gate':>5}  {'q1':>3}  {'q2':>3}  {'theta':>14}")
    print("-" * 40)

    for i, (theta, op, a, b) in enumerate(zip(params, op_codes, q1, q2), start=1):
        name = op_names.get(int(op), f"OP{op}")
        a = int(a)
        b = int(b)
        if int(op) == OP_RZZ:
            print(f"{i:4d}  {name:>5}  {a:3d}  {b:3d}  {float(theta):14.8f}")
        else:
            print(f"{i:4d}  {name:>5}  {a:3d}  {'-':>3}  {float(theta):14.8f}")


# ============================================================
# ADAPT-VQE loop
# ============================================================
def adapt_vqe(
    n_qubits,
    target,
    pool,
    max_op,
    eps_fid=1e-6,
    patience=10,
    method="L-BFGS-B",
    maxiter=200,
    grad_eps=1e-6,
):
    i0_table, i1_table = precompute_1q_indices(n_qubits)
    sign_tensor = build_sign_tensor(n_qubits)

    pool_ops, pool_q1, pool_q2 = pool_to_arrays(pool)

    params   = np.array([], dtype=np.float64)
    op_codes = np.array([], dtype=np.int32)
    q1       = np.array([], dtype=np.int32)
    q2       = np.array([], dtype=np.int32)

    fidelities = []
    delta_fidelities = []

    prev_fid = 0.0
    small_count = 0

    for i in range(max_op):
        params, op_codes, q1, q2, score = find_best_op(
            n_qubits=n_qubits,
            params=params,
            op_codes=jnp.asarray(op_codes),
            q1=jnp.asarray(q1),
            q2=jnp.asarray(q2),
            target_state=target,
            pool_ops=pool_ops,
            pool_q1=pool_q1,
            pool_q2=pool_q2,
            i0_table=i0_table,
            i1_table=i1_table,
            sign_tensor=sign_tensor,
            method=method,
            maxiter=maxiter,
            grad_eps=grad_eps,
        )

        fid = float(fidelity(
            n_qubits=n_qubits,
            params=jnp.asarray(params),
            op_codes=jnp.asarray(op_codes),
            q1=jnp.asarray(q1),
            q2=jnp.asarray(q2),
            target_state=target,
            i0_table=i0_table,
            i1_table=i1_table,
            sign_tensor=sign_tensor,
        ))

        fidelities.append(fid)
        delta = abs(fid - prev_fid)
        delta_fidelities.append(delta)

        print(
            f"step {i+1:3d}/{max_op}"
            f" | fidelity: {fid:.8f}"
            f" | Δfid: {delta:.2e}"
            f" | score: {score:.2e}"
        )

        if i > 0 and delta < eps_fid:
            small_count += 1
        else:
            small_count = 0

        if small_count >= patience:
            print(f"Stopping early: |Δfid| < {eps_fid} for {patience} consecutive steps.")
            break

        prev_fid = fid

    return params, op_codes, q1, q2, fidelities, delta_fidelities


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    n_qubits = 6#12
    max_op   = 200

    eps_fid  = 1e-6
    patience = 10

    seed = 0
    key = jax.random.PRNGKey(seed)

    pool = make_default_pool(n_qubits)
    target = random_haar_state(n_qubits, key)

    params, op_codes, q1, q2, fidelities, delta_fidelities = adapt_vqe(
        n_qubits=n_qubits,
        target=target,
        pool=pool,
        max_op=max_op,
        eps_fid=eps_fid,
        patience=patience,
        method="L-BFGS-B",
        maxiter=200,
        grad_eps=1e-6,
    )

    print("\nFinal results")
    print("Number of operators:", len(op_codes))
    print("Best fidelity:", fidelities[-1] if fidelities else 0.0)
    print("Best op_codes:", op_codes)
    print("Best q1:", q1)
    print("Best q2:", q2)

    print_circuit_summary(params, op_codes, q1, q2)

    qc = build_qiskit_circuit(n_qubits, params, op_codes, q1, q2)

    print("\nFinal circuit:")
    print(qc.draw(output="text", fold=120))

    # Optional graphical circuit plot
    # Requires: pip install qiskit pylatexenc
    try:
        fig_circ = qc.draw(output="mpl", fold=40)
        plt.show()
    except Exception as e:
        print("\nCould not generate matplotlib circuit drawing.")
        print("Reason:", e)
        print("You can still use the text circuit above.")
        print("If needed, install: pip install qiskit pylatexenc")

   # ============================================================
    # Plot: ONLY fidelity
    # ============================================================
    steps = np.arange(1, len(fidelities) + 1)
    plt.figure()
    plt.plot(steps, fidelities, linewidth=2)

    plt.xlabel("Number of operators")
    plt.ylabel("Fidelity")
    plt.title("ADAPT-VQE convergence")
    plt.grid(True, linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.show()