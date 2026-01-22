"""
JAX ADAPT-style gate program runner (Fix 1):
- No GateProgram class; pass arrays directly.
- n_qubits is static for jit.
- Supports RX/RY/RZ and RZZ.
- Qubit indexing convention: q=0 is the least-significant bit (LSB).

This file is self-contained and runnable.
"""

import jax
import jax.numpy as jnp


# -----------------------------
# Op-codes
# -----------------------------
OP_RX  = 0
OP_RY  = 1
OP_RZ  = 2
OP_RZZ = 3


# -----------------------------
# Gate matrices (2x2) for single-qubit rotations
# -----------------------------
def RX(theta):
    c = jnp.cos(theta / 2)
    s = jnp.sin(theta / 2)
    return jnp.array([[c, -1j * s],
                      [-1j * s, c]], dtype=jnp.complex64)

def RY(theta):
    c = jnp.cos(theta / 2)
    s = jnp.sin(theta / 2)
    return jnp.array([[c, -s],
                      [s,  c]], dtype=jnp.complex64)

def RZ(theta):
    return jnp.array([[jnp.exp(-1j * theta / 2), 0.0 + 0.0j],
                      [0.0 + 0.0j, jnp.exp(1j * theta / 2)]], dtype=jnp.complex64)


# -----------------------------
# Initial state |0...0>
# -----------------------------
def init_state(n_qubits: int, dtype=jnp.complex64):
    dim = 2 ** n_qubits
    psi = jnp.zeros((dim,), dtype=dtype)
    return psi.at[0].set(1.0 + 0.0j)


# -----------------------------
# Fast application routines without building 2^n x 2^n matrices
# These are JIT-friendly and work with dynamic q indices.
# -----------------------------
@jax.jit
def apply_1q_gate(psi: jnp.ndarray, U: jnp.ndarray, q: jnp.ndarray, n_qubits: int):
    """
    Apply a 2x2 gate U to qubit q (0=LSB) on an n_qubits statevector psi.

    Works with q as a traced (dynamic) integer under JIT.
    Complexity: O(2^n)
    """
    dim = psi.shape[0]
    half = dim // 2

    # indices 0..2^(n-1)-1 enumerate basis states with target bit removed
    k = jnp.arange(half, dtype=jnp.int32)

    # Split k into "low bits" below q and "high bits" above q, then insert target bit
    low_mask = (1 << q) - 1
    low = k & low_mask
    high = k >> q

    base = (high << (q + 1)) | low   # target bit is 0 here
    i0 = base
    i1 = base | (1 << q)             # target bit is 1

    a0 = psi[i0]
    a1 = psi[i1]

    # Apply 2x2 gate:
    # [b0]   [U00 U01] [a0]
    # [b1] = [U10 U11] [a1]
    b0 = U[0, 0] * a0 + U[0, 1] * a1
    b1 = U[1, 0] * a0 + U[1, 1] * a1

    psi2 = psi.at[i0].set(b0)
    psi2 = psi2.at[i1].set(b1)
    return psi2


@jax.jit
def apply_rzz(psi: jnp.ndarray, theta: jnp.ndarray, q1: jnp.ndarray, q2: jnp.ndarray, n_qubits: int):
    """
    Apply RZZ(theta) = exp(-i theta/2 * Z⊗Z) to qubits (q1, q2), 0=LSB.
    Complexity: O(2^n), implemented as an elementwise phase multiply.
    """
    dim = psi.shape[0]
    idx = jnp.arange(dim, dtype=jnp.int32)

    b1 = (idx >> q1) & 1
    b2 = (idx >> q2) & 1

    # eigenvalue s of Z⊗Z: +1 if bits equal, -1 if bits differ
    s = jnp.where(b1 == b2, 1.0, -1.0).astype(jnp.float32)

    phase = jnp.exp(-1j * theta / 2 * s).astype(psi.dtype)
    return psi * phase


# -----------------------------
# One step in the program (for scan)
# -----------------------------
def _apply_step(carry_state, step, n_qubits: int):
    theta, op, q1, q2 = step

    def do_rx(s):  return apply_1q_gate(s, RX(theta), q1, n_qubits)
    def do_ry(s):  return apply_1q_gate(s, RY(theta), q1, n_qubits)
    def do_rz(s):  return apply_1q_gate(s, RZ(theta), q1, n_qubits)
    def do_rzz(s): return apply_rzz(s, theta, q1, q2, n_qubits)

    new_state = jax.lax.switch(op, (do_rx, do_ry, do_rz, do_rzz), carry_state)
    return new_state, None


# -----------------------------
# Main runner (Fix 1): arrays in, state out
# -----------------------------
def run_program_arrays(n_qubits: int,
                       params: jnp.ndarray,
                       op_codes: jnp.ndarray,
                       q1: jnp.ndarray,
                       q2: jnp.ndarray):
    """
    Execute a program of length L, where each step i has:
      theta = params[i]
      op    = op_codes[i] in {OP_RX, OP_RY, OP_RZ, OP_RZZ}
      q1[i], q2[i] are qubit indices (q2 ignored for 1q gates)

    Returns final statevector (2**n_qubits,).
    """
    psi0 = init_state(n_qubits)

    steps = (
        params.astype(jnp.float32),
        op_codes.astype(jnp.int32),
        q1.astype(jnp.int32),
        q2.astype(jnp.int32),
    )

    psiT, _ = jax.lax.scan(lambda s, st: _apply_step(s, st, n_qubits), psi0, steps)
    return psiT


# JIT-compile with n_qubits static
run_program_jit = jax.jit(run_program_arrays, static_argnames=("n_qubits",))


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Build a tiny program:
    # RX(0.1) on qubit 0
    # RZZ(0.2) on qubits (0,1)
    # RY(0.7) on qubit 5
    params   = jnp.array([0.1, 0.2, 0.7], dtype=jnp.float32)
    op_codes = jnp.array([OP_RX, OP_RZZ, OP_RY], dtype=jnp.int32)
    q1       = jnp.array([0, 0, 5], dtype=jnp.int32)
    q2       = jnp.array([0, 1, 0], dtype=jnp.int32)  # only used for OP_RZZ

    psi = run_program_jit(n_qubits=12, params=params, op_codes=op_codes, q1=q1, q2=q2)
    loss=loss=run_program_jit(n_qubits=12, params=params, op_codes=op_codes, q1=q1, q2=q2)
    # Sanity checks
    norm = jnp.vdot(psi, psi).real
    print("Final state shape:", psi.shape)
    print("Norm:", float(norm))
    print("First 8 amplitudes:", psi[:8])
