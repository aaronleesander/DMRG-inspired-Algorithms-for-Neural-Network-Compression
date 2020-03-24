# Title: DMRG for Bachelor Thesis
# Author: Aaron Sander
# Date: March 2020

# This program is used for initial learning of tensor network methods to be
# used in my bachelor thesis.
# It is an implementation of Matrix Product States (MPS) and Density Matrix
# Renormalization Group (DMRG) for finding the ground state of an arbitrary
# Hamiltonian


######################### IMPORTS #############################################
import numpy as np


######################## CLASSES ##############################################
class MPO:
    def __init__(self, left_bound, inner, right_bound):
        # Leftmost lattice position
        self.left_bound = left_bound
        # Middle lattice positions
        self.inner = inner
        # Rightmost lattice position
        self.right_bound = right_bound


class MPS:
    def __init__(self, left_bound, inner, right_bound):
        self.left_bound = left_bound
        self.inner = inner
        self.right_bound = right_bound


###################### FUNCTIONS ##############################################
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
        if dir == 'right' or 'left':  # Seems to be direction-independent
            tensor = np.einsum('ijk, iab->jakb', A, B)
        # Reshape collapses indices to (j*a, k*b)
        tensor = np.reshape(tensor, (A.shape[1]*B.shape[1], A.shape[2]*B.shape[2]))

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
N = 3

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

MPO = MPO(left_bound, inner, right_bound)


######################### INITIALIZATION MPS ##################################
# We initialize each wavefunction matrix as:
# A_1_ket = ( |+>    A_i_ket = ( |+X+|      A_N_ket = (|+> |->)
#             |-> )              |-x-| )

# Bond Dimension
d = 5

up_ket = np.zeros((d, 1))
up_ket[0,0] = 1
down_ket = np.zeros((d, 1))
down_ket[1,0] = 1

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

MPS_bra = MPS(A_1_dagger, A_i_dagger, A_N_dagger)
MPS_ket = MPS(A_1, A_i, A_N)

# TODO: Find a better notation than using MPS_bra, MPS_ket
# TODO: MPS should not have left_bound, inner, right_bound because
#       all inner tensors need to be modified, unlike MPO where
#       all inner tensors remain the same.
# TODO: Verify if A_i == A_i_dagger is correct
# NOTE: May be necessary to reshape wavefunctions to D x D x 2


######################## TESTING ##############################################
##################### CONTRACT HAMILTONIAN L->R ###############################
# Initialize with first lattice position
tensor = MPO.left_bound
# Loop over all the inner lattice positions
for i in range(2, N):
    tensor = contract_horizontal(tensor, MPO.inner, "right")
# Final lattice position has different indices so is done alone
E_L = contract_horizontal(tensor, MPO.right_bound, "right")


##################### CONTRACT HAMILTONIAN R->L ###############################
# Initialize with first lattice position
tensor = MPO.right_bound
# Loop over all the inner lattice positions
for i in range(2, N):
    tensor = contract_horizontal(tensor, MPO.inner, "left")
# Final lattice position has different indices so is done alone
E_R = contract_horizontal(tensor, MPO.left_bound, "left")

if E_L.all() == E_R.all():
    print("Hamiltonian contracts properly in both directions")


####################### CONTRACT FIRST LATTICE POSITION DOWNWARDS #############
# Dimensions (3*d x 2)
first = contract_vertical(MPS_bra.left_bound, MPO.left_bound, 'down')
# Dimensions (3*d*d)
pos1_contract_down = contract_vertical(first, MPS_ket.left_bound, 'down')


####################### CONTRACT FIRST LATTICE POSITION UPWARDS ###############
# Dimensions (3*d x 2)
first = contract_vertical(MPS_ket.left_bound, MPO.left_bound, 'up')
# Dimensions (3*d*d)
pos1_contract_up = contract_vertical(first, MPS_bra.left_bound, 'up')

if pos1_contract_down.all() == pos1_contract_up.all():
    print("Left lattice position contracts properly in both directions")


################## CONTRACT INNER LATTICE POSITION DOWNARDS ###################
# Dimensions (3d x 3d x 2)
first = contract_vertical(MPS_bra.inner, MPO.inner, 'down')
# Dimensions
posI_contract_down = contract_vertical(first, MPS_ket.inner, 'down')

################## CONTRACT INNER LATTICE POSITION UPWARDS ####################
# Dimensions (3d x 3d x 2)
first = contract_vertical(MPS_ket.inner, MPO.inner, 'up')
# Dimensions (3*d*d)
posI_contract_up = contract_vertical(first, MPS_bra.inner, 'up')

if posI_contract_down.all() == posI_contract_up.all():
    print("Inner lattice position contracts properly in both directions")


################## CONTRACT LAST LATTICE POSITION DOWNWARDS #################
# Dimensions (3d x 2)
first = contract_vertical(MPS_bra.right_bound, MPO.right_bound, 'down')
# Dimensions (3*d*d)
posN_contract_down = contract_vertical(first, MPS_ket.right_bound, 'down')

################## CONTRACT LAST LATTICE POSITION UPWARDS #################
# Dimensions (3d x 2)
first = contract_vertical(MPS_ket.right_bound, MPO.right_bound, 'up')
# Dimensions (3*d*d)
posN_contract_up = contract_vertical(first, MPS_bra.right_bound, 'up')

if posN_contract_down.all() == posN_contract_up.all():
    print("Right lattice position contracts properly in both directions")
