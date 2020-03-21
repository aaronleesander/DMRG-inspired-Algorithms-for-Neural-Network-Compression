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

H = Hamiltonian()