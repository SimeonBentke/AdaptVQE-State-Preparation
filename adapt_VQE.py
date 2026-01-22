import numpy as np
from itertools import product

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from qiskit_aer.primitives import EstimatorV2
from qiskit_algorithms.optimizers import SLSQP
from qiskit_algorithms.minimum_eigensolvers import VQE, AdaptVQE


# ----------------------------
# Pool: {X_i, Y_i} ∪ {Z_i Z_j}
# ----------------------------
def pauli_string(n, mapping):
    s = ["I"] * n
    for i, p in mapping.items():
        s[i] = p
    return "".join(s)


def native_pool(n, edges):
    pool = []
    for i in range(n):
        pool.append(SparsePauliOp.from_list([(pauli_string(n, {i: "X"}), 1.0)]))
        pool.append(SparsePauliOp.from_list([(pauli_string(n, {i: "Y"}), 1.0)]))
    for (i, j) in edges:
        pool.append(SparsePauliOp.from_list([(pauli_string(n, {i: "Z", j: "Z"}), 1.0)]))
    return pool


# -----------------------------------------
# Build H = I - |psi><psi| as Pauli expansion
# H = sum_{P in Paulis} c_P P,  with  c_P = Tr(P H)/2^n
# -----------------------------------------
_PAULI_MATS = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def kron_pauli(label: str) -> np.ndarray:
    m = _PAULI_MATS[label[0]]
    for ch in label[1:]:
        m = np.kron(m, _PAULI_MATS[ch])
    return m


def projector_hamiltonian_pauli_sum(psi: np.ndarray, tol: float = 1e-12) -> SparsePauliOp:
    """
    Returns SparsePauliOp for H = I - |psi><psi| as a Pauli sum.
    Scaling: O(4^n * 2^(2n)) if done naively; okay for small n (e.g., n<=6).
    """
    psi = np.asarray(psi, dtype=complex)
    dim = psi.shape[0]
    n = int(np.log2(dim))
    if 2**n != dim:
        raise ValueError("psi length must be a power of 2")

    rho = np.outer(psi, psi.conj())  # |psi><psi|
    H = np.eye(dim, dtype=complex) - rho

    terms = []
    denom = 2**n

    for label in map("".join, product("IXYZ", repeat=n)):
        P = kron_pauli(label)
        coeff = np.trace(P @ H) / denom  # should be real for Hermitian H
        coeff = float(np.real_if_close(coeff))
        if abs(coeff) > tol:
            terms.append((label, coeff))

    return SparsePauliOp.from_list(terms)


# ----------------------------
# Main
# ----------------------------
def main():
    # Choose a small n (projector expansion scales as 4^n)
    n = 4
    edges = [(i, i + 1) for i in range(n - 1)]  # line connectivity; change if needed

    # Random target state
    seed = 7
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    vec = vec / np.linalg.norm(vec)
    psi_tgt = Statevector(vec)

    # Hamiltonian H = I - |psi_tgt><psi_tgt|
    H = projector_hamiltonian_pauli_sum(psi_tgt.data)

    # Pool
    pool_ops = native_pool(n, edges)

    # Ansatz: start from |0...0>
    ansatz0 = QuantumCircuit(n)

    # VQE + AdaptVQE
    estimator = EstimatorV2()
    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz0,
        optimizer=SLSQP(maxiter=200),
        initial_point=np.array([]),
    )

    # Construct AdaptVQE (pool-enabled versions accept operators in the constructor;
    # we keep a robust constructor wrapper)
    try:
        adapt = AdaptVQE(vqe_solver=vqe, operators=pool_ops)
    except TypeError:
        try:
            adapt = AdaptVQE(vqe, operators=pool_ops)
        except TypeError:
            adapt = AdaptVQE(vqe, pool_ops)

    # Tuning
    if hasattr(adapt, "gradient_threshold"):
        adapt.gradient_threshold = 1e-6
    if hasattr(adapt, "max_iterations"):
        adapt.max_iterations = 50

    res = adapt.compute_minimum_eigenvalue(H)

    # Extract final circuit
    final_ansatz = getattr(res, "ansatz", None) or vqe.ansatz
    params = getattr(res, "optimal_point", None)
    if params is None:
        params = np.array([])

    final_circuit = final_ansatz.assign_parameters(params)
    psi_final = Statevector.from_instruction(final_circuit)

    fidelity = float(abs(np.vdot(psi_tgt.data, psi_final.data)) ** 2)
    energy = float(np.real(res.eigenvalue))

    print("\n=== Projector state prep with ADAPT-VQE ===")
    print(f"n = {n}, seed = {seed}, pool size = {len(pool_ops)}")
    print(f"Final energy <H>      = {energy:.12f}")
    print(f"Final fidelity F      = {fidelity:.12f}")
    print(f"Check (1 - F)         = {1 - fidelity:.12f}")
    print("\nFinal circuit:")
    print(final_circuit.draw("text"))


if __name__ == "__main__":
    main()
