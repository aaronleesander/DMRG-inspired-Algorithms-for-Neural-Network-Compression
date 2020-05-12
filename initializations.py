######################## INITIALIZATIONS ######################################

import numpy as np


def initialize_random_state(num_particles, bond_dim, phys_dim):
    # Dimensions (phys_dim x d)
    M_1 = np.random.rand(phys_dim, bond_dim)

    # Dimensions (d x d x phys_dim)
    M_i = np.random.rand(bond_dim, bond_dim, phys_dim)

    # Dimensions (phys_dim x d)
    M_N = np.random.rand(phys_dim, bond_dim)

    MPS = [M_1] + [M_i]*(num_particles-2) + [M_N]
    return MPS


def initialize_W_state(num_particles):
    # d = 2
    # Dimensions (2 x d)
    M_1 = np.array([np.array([1, 0]),
                    np.array([0, 1])])

    # Dimensions (d x d x 2)
    # (2 x d x d) -> (d x d x 2)
    M_i = np.array([np.array([[1, 0],
                              [0, 1]]),
                    np.array([[0, 1],
                              [0, 0]])])
    M_i = np.transpose(M_i, (1, 2, 0))

    # Dimensions (2 x d)
    M_N = np.array([np.array([[0],
                              [1]]),
                    np.array([[1],
                              [0]])])
    M_N = np.squeeze(M_N)

    MPS = [M_1] + [M_i]*(num_particles-2) + [M_N]
    return MPS


def initialize_GHZ_state(num_particles):
    # d = 2
    # Dimensions (2 x d)
    M_1 = np.array([[1, 0], [0, 1]])

    # Dimensions (d x d x 2)
    # (2 x d x d) -> (d x d x 2)
    M_i = np.array([np.array([[1, 0],
                              [0, 0]]),
                    np.array([[0, 0],
                              [0, 1]])])
    M_i = np.transpose(M_i, (1, 2, 0))

    # Dimensions (2 x d)
    M_N = np.array([np.array([[1],
                              [0]]),
                    np.array([[0],
                              [1]])])
    M_N = np.squeeze(M_N)

    MPS = [M_1] + [M_i]*(num_particles-2) + [M_N]
    return MPS


### Quantum Ising Model ###
def initialize_quantum_ising(num_particles):
    # Operators
    pauli_z = np.array([[1, 0],
                        [0, -1]])

    pauli_x = np.array([[0, 1],
                        [1, 0]])

    zero = np.zeros((2, 2))
    identity = np.identity(2)

    # Interaction parameter
    g = 2

    # Initialization of Hamiltonian MPO (entries done by hand)
    # Dimensions (1x3x2x2)->(3x2x2)
    left_bound = np.array([identity, pauli_z, g*pauli_x])

    # Dimensions (3x3x2x2)
    inner = np.array([np.array([identity, pauli_z, g*pauli_x]),
                      np.array([zero, zero, pauli_z]),
                      np.array([zero, zero, np.identity(2)])])

    # Dimensions (3x1x2x2)->3x2x2
    right_bound = np.array([[g*pauli_x],
                            [pauli_z],
                            [identity]])
    right_bound = np.squeeze(right_bound)  # Removes unnecessary index

    MPO = [left_bound] + [inner]*(num_particles-2) + [right_bound]

    return MPO
