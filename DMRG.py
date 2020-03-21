# Title: DMRG for Bachelor Thesis
# Author: Aaron Sander
# Date: March 2020

# This program is used for initial learning of tensor network methods to be used
# in my bachelor thesis.
# It is an implementation of Matrix Product States (MPS) and Density Matrix
# Renormalization Group (DMRG) for finding the ground state of an arbitrary
# Hamiltonian

import numpy as np

# Initialization
pauli_z = np.array([[1,0],
                    [0,-1]])

pauli_x = np.array([[0,1],
                   [1,0]])

zero = np.zeros((2,2))

identity = np.identity(2)

g = 1

class Hamiltonian:
    # Matrix Product States
    # First lattice position (1x3x2x2)
    left_bound = np.array([identity, pauli_z, g*pauli_x])
    left_bound = left_bound[np.newaxis, :] # Reshape to add the first index
    
    # Middle lattice positions (3x3x2x2)
    middle = np.array([np.array([identity, pauli_z, g*pauli_x]),
                       np.array([zero, zero, pauli_z]),
                       np.array([zero, zero, np.identity(2)])])

    # Last lattice position (3x1x2x2)
    right_bound = np.array([[g*pauli_x],
                               [pauli_z],
                               [identity]])

def contract_from_left(A, B):
    tensor = np.einsum('ijk,aibc->ajbkc', A, B)
    tensor = np.reshape(tensor, (3,4,4)) # Collapses indices to (c, a*j, b*k)
                                         # TODO: Verify correct reshape
    return tensor

H = Hamiltonian()

first_collapse = np.einsum('ijk, aibc->ajbkc', H.left_bound, H.middle)
first_collapse = np.reshape(first_collapse, (3,4,4))

N = 3
inner_tensor = first_collapse
for i in range(1, N-1):
    inner_tensor = np.einsum('ijk, aibc->ajbkc', inner_tensor, H.middle)
    inner_tensor = np.reshape(inner_tensor, (3,2,2))

last_collapse = np.einsum('ijk,iab->jkab', inner_tensor, H.right_bound)
last_collapse = np.reshape(last_collapse, (2^N,2^N))