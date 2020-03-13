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

g = 1

# Matrix Product States
# First lattice position (1x3x2x2)
A_1 = np.array([np.identity(2), pauli_z, g*pauli_x])
A_1 = A_1[np.newaxis, :] # Reshape to add the first index

# Last lattice position (3x1x2x2)
A_N = np.array([[g*pauli_x],
               [pauli_z],
               [np.identity(2)]])

# Middle lattice positions (3x3x2x2)
A_i = np.array([[np.identity(2), pauli_z, g*pauli_x],
                [0, 0, pauli_z],
                [0, 0, np.identity(2)]])