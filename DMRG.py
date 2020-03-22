# Title: DMRG for Bachelor Thesis
# Author: Aaron Sander
# Date: March 2020

# This program is used for initial learning of tensor network methods to be used
# in my bachelor thesis.
# It is an implementation of Matrix Product States (MPS) and Density Matrix
# Renormalization Group (DMRG) for finding the ground state of an arbitrary
# Hamiltonian

######################### IMPORTS ##############################################
import numpy as np

######################## CLASSES ###############################################
class Hamiltonian:
    def __init__(self, left_bound, inner, right_bound):
        # Leftmost lattice position 
        self.left_bound = left_bound
        # Middle lattice positions 
        self.inner = inner
        # Rightmost lattice position 
        self.right_bound = right_bound


### TODO: Define Wavefunction class

###################### FUNCTIONS ##############################################
# Used to contract Hamiltonian horizontally from left to right
def contract_left_to_right(A, B, pos):
    if B.ndim == 4: # Inner lattice positions
        tensor = np.einsum('ijk,iabc->ajbkc', A, B)
        # Reshape collapses indices to (a, j*b, k*c) for i particles
        tensor = np.reshape(tensor, (3,2**pos,2**pos))

    if B.ndim == 3: # Final lattice position
        tensor = np.einsum('ijk,iab->jakb', A, B)
        # Reshape collapses indices to (j*a, k*b) for i particles
        tensor = np.reshape(tensor, (2**pos,2**pos))

    return tensor
# TODO: When collapsing with a wavefunction
#       the tensor shape will be based on size of
#       wavefunction matrices also, not just lattice pos
# TODO: Verify correct reshape
# NOTE: In DMRG, the dimension of the final lattice position
#       will be a good marker for when to reverse direction

##################### INITIALIZATION ###########################################
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
# Done using Matrix Product States by hand
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
right_bound = np.squeeze(right_bound) # Removes unnecessary index 

H = Hamiltonian(left_bound, inner, right_bound)

##################### CONTRACT HAMILTONIAN L->R ################################
# Initialize with first lattice position
tensor = H.left_bound
# Loop over all the inner lattice positions
for i in range(2, N):
     tensor = contract_left_to_right(tensor, H.inner, i)
# Final lattice position has different indices so is done alone
E = contract_left_to_right(tensor, H.right_bound, N)