import numpy as np

import canonical_forms as can


def initialize_random_normed_vector(length):
    """ Initializes a normed vector of a given length
    Args:
        length: Number of elements (Ex. Number of bits in data vector)

    Returns:
        vector: Normed vector with given length
    """
    vector = np.random.rand(length)
    norm = np.linalg.norm(vector)
    vector = vector / norm
    return vector


def initialize_random_normed_state_MPS(num_sites, bond_dim, phys_dim):
    """ Initializes a Matrix Product State containing random values
    Args:
        num_sites: Number of tensors in MPS
        bond_dim: Virtual dimension between each tensor
        phys_dim: Physical dimension

    Returns:
        MPS: List of tensors of length num_sites
             Left Bound MPS[1] has shape (phys_dim x right_bond)
             Inner MPS[i] has shape (left_bond x right_bond x phys_dim)
             Right Bound MPS[-1] has shape (phys_dim x left_bond)
    """
    M_1 = np.random.rand(phys_dim, bond_dim)
    M_i = np.random.rand(bond_dim, bond_dim, phys_dim)
    M_N = np.random.rand(phys_dim, bond_dim)

    MPS = [M_1] + [M_i]*(num_sites-2) + [M_N]
    MPS, _ = can.left_normalize(MPS)
    MPS, _ = can.right_normalize(MPS)
    return MPS


def initialize_random_MPS_with_changing_phys_dim(phys_dim, num_sites, bond_dim):
    """ Initializes and MPS with different physical dimensions at each site

    Args:
        phys_dim: List of physical dimensions by site
        num_sites: Integer of total number of sites
        bond_dim: Integer of required bond dimensions

    Returns:
        MPS: List of left and right normalized
             tensors of MPS with given physical and bond dimensions
    """
    MPS = []
    M_1 = np.random.rand(phys_dim[0], bond_dim)
    MPS.append(M_1)
    for i in range(1, num_sites-1):
        M_i = np.random.rand(bond_dim, bond_dim, phys_dim[i])
        MPS.append(M_i)
    M_N = np.random.rand(phys_dim[-1], bond_dim)
    MPS.append(M_N)

    MPS, _ = can.left_normalize(MPS)
    MPS, _ = can.right_normalize(MPS)
    return MPS

def initialize_W_state_MPS(num_sites):
    """ Initializes the W-state as a Matrix Product State
    Args:
        num_sites: Number of tensors in MPS

    Returns:
        MPS: List of tensors of length num_sites
             Left Bound MPS[1] has shape (phys_dim x right_bond)
             Inner MPS[i] has shape (left_bond x right_bond x phys_dim)
             Right Bound MPS[-1] has shape (phys_dim x left_bond)

             Initialization done by hand
    """
    M_1 = np.array([np.array([1, 0]),
                    np.array([0, 1])])

    M_i = np.array([np.array([[1, 0],
                              [0, 1]]),
                    np.array([[0, 1],
                              [0, 0]])])
    M_i = np.transpose(M_i, (1, 2, 0))

    M_N = np.array([np.array([[0],
                              [1]]),
                    np.array([[1],
                              [0]])])
    M_N = np.squeeze(M_N)  # Removes dummy index

    MPS = [M_1] + [M_i]*(num_sites-2) + [M_N]
    return MPS


def initialize_GHZ_state_MPS(num_sites):
    """ Initializes the GHZ-state as a Matrix Product State
    Args:
        num_sites: Number of tensors in MPS

    Returns:
        MPS: List of tensors of length num_sites
             Left Bound MPS[1] has shape (phys_dim x right_bond)
             Inner MPS[i] has shape (left_bond x right_bond x phys_dim)
             Right Bound MPS[-1] has shape (phys_dim x left_bond)

             Initialization done by hand
    """
    M_1 = np.array([[1, 0], [0, 1]])

    M_i = np.array([np.array([[1, 0],
                              [0, 0]]),
                    np.array([[0, 0],
                              [0, 1]])])
    M_i = np.transpose(M_i, (1, 2, 0))

    M_N = np.array([np.array([[1],
                              [0]]),
                    np.array([[0],
                              [1]])])
    M_N = np.squeeze(M_N)  # Removes dummy index

    MPS = [M_1] + [M_i]*(num_sites-2) + [M_N]
    return MPS


def initialize_random_MPO(num_sites, bond_dim, phys_dim):
    """ Initializes a Matrix Product State containing random values
    Args:
        num_sites: Number of tensors in MPS
        bond_dim: Virtual dimension between each tensor
        phys_dim: Physical dimension

    Returns:
        MPO: List of tensors of length num_sites

             Left Bound MPO[1] has shape (right_bond
                                         x lower_phys_dim
                                         x upper_phys_dim)
             Inner MPO[i] has shape (left_bond
                                    x right_bond
                                    x lower_phys_dim
                                    x upper_phys_dim)
            Right Bound MPO[N] has shape (right_bond
                                         x lower_phys_dim
                                         x upper_phys_dim)
    """
    M_1 = np.random.rand(bond_dim, phys_dim, phys_dim)
    M_i = np.random.rand(bond_dim, bond_dim, phys_dim, phys_dim)
    M_N = np.random.rand(bond_dim, phys_dim, phys_dim)

    MPO = [M_1] + [M_i]*(num_sites-2) + [M_N]
    return MPO


def initialize_quantum_ising_MPO(num_sites, J, g):
    """ Initializes the Quantum Ising Model as a Matrix Product Operator
    Args:
        num_sites: Number of tensors in MPO
        g: Interaction parameter
        J: Interaction type, attached to first pauli_z in MPO

    Returns:
        MPO: List of tensors of length num_sites

             Left Bound MPO[1] has shape (right_bond
                                         x lower_phys_dim
                                         x upper_phys_dim)
             Inner MPO[i] has shape (left_bond
                                    x right_bond
                                    x lower_phys_dim
                                    x upper_phys_dim)
             Right Bound MPO[N] has shape (right_bond
                                         x lower_phys_dim
                                         x upper_phys_dim)

             Initialization done by hand
    """
    pauli_z = np.array([[1, 0],
                        [0, -1]])

    pauli_x = np.array([[0, 1],
                        [1, 0]])
    zero = np.zeros((2, 2))
    identity = np.identity(2)

    left_bound = np.array([identity, -J*pauli_z, -g*pauli_x])

    inner = np.array([np.array([identity, -J*pauli_z, -g*pauli_x]),
                      np.array([zero, zero, pauli_z]),
                      np.array([zero, zero, identity])])

    right_bound = np.array([[-g*pauli_x],
                            [pauli_z],
                            [identity]])
    right_bound = np.squeeze(right_bound)  # Removes dummy index

    MPO = [left_bound] + [inner]*(num_sites-2) + [right_bound]
    return MPO


def initialize_gate_MPO(num_sites):
    """ Initializes the Quantum Ising Model as a Matrix Product Operator
    Args:
        num_sites: Number of tensors in MPO
        g: Interaction parameter
        J: Interaction type, attached to first pauli_z in MPO

    Returns:
        MPO: List of tensors of length num_sites

             Left Bound MPO[1] has shape (right_bond
                                         x lower_phys_dim
                                         x upper_phys_dim)
             Inner MPO[i] has shape (left_bond
                                    x right_bond
                                    x lower_phys_dim
                                    x upper_phys_dim)
            Right Bound MPO[N] has shape (right_bond
                                         x lower_phys_dim
                                         x upper_phys_dim)

             Initialization done by hand
    """
    XOR = np.array([[0, 1],
                    [1, 0]])

    AND = np.array([[0, 0],
                    [0, 1]])

    zero = np.zeros((2, 2))
    identity = np.identity(2)

    left_bound = np.array([identity, XOR, zero])

    inner = np.array([np.array([identity, XOR, zero]),
                      np.array([zero, zero, AND]),
                      np.array([zero, zero, zero])])

    right_bound = np.array([[identity],
                            [AND],
                            [identity]])
    right_bound = np.squeeze(right_bound)  # Removes dummy index

    MPO = [left_bound] + [inner]*(num_sites-2) + [right_bound]
    return MPO
