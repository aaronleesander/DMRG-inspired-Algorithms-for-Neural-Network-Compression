# Title: DMRG for Bachelor Thesis
# Author: Aaron Sander
# Date: March 2020

# This program is used for initial learning of tensor network methods
# to be used in my bachelor thesis.
# It is an implementation of Matrix Product States (MPS) and Density Matrix
# Renormalization Group (DMRG) for finding the ground state of an arbitrary
# Hamiltonian


######################### IMPORTS #############################################
import numpy as np


######################## CLASSES ##############################################
class matrixProductOperator:
    def __init__(self, left_bound, inner, right_bound):
        # Leftmost lattice position
        self.left_bound = left_bound
        # Middle lattice positions
        self.inner = inner
        # Rightmost lattice position
        self.right_bound = right_bound


class matrixProductState:
    def __init__(self, left_bound, inner, right_bound):
        self.left_bound = left_bound
        self.inner = inner
        self.right_bound = right_bound


###################### FUNCTIONS ##############################################
# NOTE: All directions of contractions have been tested and work correctly.
#       This was verified by checking if result is equal for going
#       left<->right and up<->down.
# Used to contract Hamiltonian horizontally from A->B
def contract_horizontal(A, B, dir):
    if B.ndim == 4:  # Applied to inner lattice position
        if dir == 'right':
            tensor = np.einsum('ijk, iabc->ajbkc', A, B)
        if dir == 'left':  # Can be replaced with else, if for readability
            tensor = np.einsum('ijk, aibc->ajbkc', A, B)
        # Reshape collapses indices to (a, j*b, k*c)
        tensor = np.reshape(tensor, (A.shape[0], A.shape[1]*B.shape[2], A.shape[2]*B.shape[3]))

    if B.ndim == 3:  # Applied to outer lattice position
        if dir == 'right' or 'left':  # Can be removed, left for readability
            tensor = np.einsum('ijk, iab->jakb', A, B)
        # Reshape collapses indices to (j*a, k*b)
        tensor = np.reshape(tensor, (A.shape[1]*B.shape[1], A.shape[2]*B.shape[2]))

    if B.ndim == 2:
        if dir == 'right':
            tensor = np.einsum('i, ia->a', A, B)
        if dir == 'left':
            tensor = np.einsum('i, ai->a', A, B)

    if B.ndim == 1: # Inner Product
        if dir == 'right' or 'left':  # Can be removed, left for readability
            tensor = np.einsum('i, i', A, B)

    return tensor


def contract_vertical(A, B, dir):
    if B.ndim == 4:  # Applied to inner tensor of Hamiltonian
        if dir == 'down' or 'up':  # NOTE: This is only correct if A_i == A_i_dagger
            tensor = np.einsum('ijk, abci->ajbkc', A, B)
            # Reshape to (j*a, k*b, c)
            tensor = np.reshape(tensor, (A.shape[1]*B.shape[0], A.shape[2]*B.shape[1], B.shape[2]))

    if B.ndim == 3 and A.ndim == 2:  #  Applied to outer tensor of Hamiltonian
        if dir == 'down':  # From Bra->Operator->Ket
            tensor = np.einsum('ij, abj->iab', A, B)
            # Reshape to (i*a, b)
            tensor = np.reshape(tensor, (A.shape[0]*B.shape[0], B.shape[1]))
        if dir == 'up':  # From Ket->Operator->Bra
            tensor = np.einsum('ij, aib->jab', A, B)
            # Reshape to (j*a, b)
            tensor = np.reshape(tensor, (A.shape[1]*B.shape[0], B.shape[2]))

    if B.ndim == 3 and A.ndim == 3:  # Applied to inner tensor of wavefunction
        if dir == 'down' or 'up':
            tensor = np.einsum('ijk, kab->aibj', A, B)
            # Reshape to (i*a, j*b)
            tensor = np.reshape(tensor, (A.shape[0]*B.shape[1], A.shape[1]*B.shape[2]))

    if B.ndim == 2:  # Applied to outer tensor of wavefunction
        if dir == 'down':
            tensor = np.einsum('ij, ja->ia', A, B)
            # Reshape to (i*a)
            tensor = np.reshape(tensor, (A.shape[0]*B.shape[1]))
        if dir == 'up':
            tensor = np.einsum('ij, aj->ia', A, B)
            # Reshape to (i*a)
            tensor = np.reshape(tensor, (A.shape[0]*B.shape[0]))

    return tensor


def calculateExpectation(MPS_bra, MPO, MPS_ket, vert_dir, horiz_dir):
    # Initialize list of tensors
    tensor = [None]*N

    # Contract <MPS|MPO|MPS> at each lattice position
    # Down: Bra -> MPO -> Ket
    # Up: Ket -> MPO -> Bra
    for i in range(0, N):
        if vert_dir == 'down':
            first_contraction = contract_vertical(MPS_bra[i], MPO[i], vert_dir)
            tensor[i] = contract_vertical(first_contraction, MPS_ket[i], vert_dir)
        if vert_dir == 'up':
            first_contraction = contract_vertical(MPS_ket[i], MPO[i], vert_dir)
            tensor[i] = contract_vertical(first_contraction, MPS_bra[i], vert_dir)

    # Contract each tensor created from above
    # Left and right necessary for scanning in DMRG
    if horiz_dir == 'right':
        E = tensor[0]
        for i in range(1, len(tensor)):
            E = contract_horizontal(E, tensor[i], horiz_dir)
    if horiz_dir == 'left':
        E = tensor[-1]
        for i in range(len(tensor)-2, -1, -1):
            E = contract_horizontal(E, tensor[i], horiz_dir)

    return E

# TODO: Verify correct reshape
# TODO: Determine if vertical/horizontal can be generalized and combined
# TODO: Use tensordot for better or standardize indices for better readability
# NOTE: In DMRG, the dimension of the final lattice position
#       will be a good marker for when to reverse direction


##################### INITIALIZATION MPO ######################################
# Operators
pauli_z = np.array([[1, 0],
                    [0, -1]])

pauli_x = np.array([[0, 1],
                    [1, 0]])

zero = np.zeros((2, 2))
identity = np.identity(2)
# Interaction parameter
g = 1
# Particles (Total lattice positions)
N = 15

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

MPO_initial = matrixProductOperator(left_bound, inner, right_bound)
MPO = [MPO_initial.left_bound] + [MPO_initial.inner]*(N-2) + [MPO_initial.right_bound]
# NOTE: The MPO never gets modified, so for a possible performance boost
#       we could just use the MPO_initial. For readability by indexing over
#       each position, we will expand it anyway so that it matches the MPS.


######################### INITIALIZATION MPS ##################################
# We initialize each wavefunction matrix as:
# A_1_ket = ( |+>    A_i_ket = ( |+X+|      A_N_ket = (|+> |->)
#             |-> )              |-x-| )

# Bond Dimension
d = 2

up_ket = np.zeros((d, 1))
up_ket[0, 0] = 1
down_ket = np.zeros((d, 1))
down_ket[1, 0] = 1

# Dimensions (2 x 1 x d) - > (2 x d)
A_1 = np.array([[up_ket],
                [down_ket]])
A_1 = np.squeeze(A_1)
# Dimensions (d x 2)
A_1_dagger = np.array(np.matrix(A_1).H)  # Only way to do Hermitian conjugate
A_1_dagger = np.squeeze(A_1_dagger)

# Dimensions (2 x 1 x d x d) -> (2 x d x d)
A_i = np.array([[np.outer(up_ket, up_ket)],  # Cannot do Hermitian conjugate of tensor
                [np.outer(down_ket, down_ket)]])
A_i = np.squeeze(A_i)
# Dimensions (1 x 2 x d x d) -> (2 x d x d)
A_i_dagger = np.array([[np.outer(up_ket, up_ket), np.outer(down_ket, down_ket)]])
A_i_dagger = np.squeeze(A_i_dagger)

# Dimensions (1 x 2 x d) -> (2 x d)
A_N = np.array([up_ket, down_ket])
A_N = np.squeeze(A_N)
# Dimensions (2 x 1 x d) -> (2 x d)
A_N_dagger = np.array(np.matrix(A_N).H)
A_N_dagger = np.squeeze(A_N_dagger)

# Initialization of MPS elements
MPS_ket_initial = matrixProductState(A_1, A_i, A_N)
MPS_bra_initial = matrixProductState(A_1_dagger, A_i_dagger, A_N_dagger)
# The inner matrices will not stay the same so we a matrix for each lattice position
MPS_ket = [MPS_ket_initial.left_bound] + [MPS_ket_initial.inner]*(N-2) + [MPS_ket_initial.right_bound]
MPS_bra = [MPS_bra_initial.left_bound] + [MPS_bra_initial.inner]*(N-2) + [MPS_bra_initial.right_bound]

# TODO: Verify if A_i == A_i_dagger is correct aka if squeeze is necessary/working correctly
# NOTE: May be necessary to reshape wavefunctions to D x D x 2

######################### TESTING #############################################
E_R = calculateExpectation(MPS_bra, MPO, MPS_ket, 'down', 'right')
E_L = calculateExpectation(MPS_bra, MPO, MPS_ket, 'down', 'left')
