# Title: DMRG for Bachelor Thesis
# Author: Aaron Sander
# Date: March 2020

# This program is used for initial learning of tensor network methods to be used
# in my bachelor thesis.
# It is an implementation of Matrix Product States (MPS) and Density Matrix
# Renormalization Group (DMRG) for finding the ground state of an arbitrary
# Hamiltonian

import numpy as np

class Hamiltonian:

    def __init__(self, left_bound, inner, right_bound):
        # First lattice position (1x3x2x2)->(3x2x2)
        self.left_bound = left_bound
        # Middle lattice positions (3x3x2x2)
        self.inner = inner
        # Last lattice position (3x1x2x2)->3x2x2
        self.right_bound = right_bound


### TODO: Define Wavefunction class

def contract_from_left(A, B, pos):
    if B.ndim == 4: # Inner lattice positions
        tensor = np.einsum('ijk,aibc->ajbkc', A, B)
        tensor = np.reshape(tensor, (3,2**pos,2**pos)) # Collapses indices to (c, a*j, b*k) for i particles
                                                       # TODO: Verify correct reshape
    if B.ndim == 3: # Final lattice position
        tensor = np.einsum('ijk,iab->jkab', A, B)
        tensor = np.reshape(tensor, (2**pos,2**pos))

    return tensor

# Initialization
pauli_z = np.array([[1,0],
                    [0,-1]])

pauli_x = np.array([[0,1],
                   [1,0]])

zero = np.zeros((2,2))
identity = np.identity(2)
# Interaction parameter
g = 1
# Lattice positions
N = 10

# Initialization of Hamiltonian
left_bound = np.array([identity, pauli_z, g*pauli_x])
    
inner = np.array([np.array([identity, pauli_z, g*pauli_x]),
                   np.array([zero, zero, pauli_z]),
                   np.array([zero, zero, np.identity(2)])])

right_bound = np.array([[g*pauli_x],
                        [pauli_z],
                        [identity]])
right_bound = np.squeeze(right_bound) # Removes unnecessary index 

H = Hamiltonian(left_bound, inner, right_bound)

# Initialize with first lattice position
tensor = H.left_bound
# Loop over all the inner lattice positions
for i in range(2, N):
     tensor = contract_from_left(tensor, H.inner, i)
# Final lattice position has different indices so is done alone
E = contract_from_left(tensor, H.right_bound, N)