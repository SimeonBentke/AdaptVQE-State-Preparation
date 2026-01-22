"""
NumPy ADAPT-style gate program runner
- No classes; pass arrays directly.
- Supports RX / RY / RZ and RZZ
- Qubit indexing: q = 0 is least-significant bit (LSB)

This file is self-contained and runnable.
"""

import numpy as np


# -----------------------------
# Op-codes
# -----------------------------
OP_RX  = 0
OP_RY  = 1
OP_RZ  = 2
OP_RZZ = 3


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
def run_program_arrays(n_qubits,
                       params,
                       op_codes,
                       q1,
                       q2):
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


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    params   = np.array([0.1, 0.2, 0.7], dtype=float)
    op_codes = np.array([OP_RX, OP_RZZ, OP_RY], dtype=int)
    q1       = np.array([0, 0, 5], dtype=int)
    q2       = np.array([0, 1, 0], dtype=int)

    psi = run_program_arrays(
        n_qubits=12,
        params=params,
        op_codes=op_codes,
        q1=q1,
        q2=q2
    )

    norm = np.vdot(psi, psi).real
    print("Final state shape:", psi.shape)
    print("Norm:", norm)
    print("First 8 amplitudes:", psi[:8])
