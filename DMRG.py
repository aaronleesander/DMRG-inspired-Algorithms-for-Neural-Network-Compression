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
# Used to contract tensors horizontally from A->B
# All 2 dimensional tensors have dimensions (ij) or (ab)
# All 3 dimensional tensors have dimensions (ijk) or (abc)
# All 4 dimensional tensors have dimensions (ijkl) or (abcd)

# Examples:
# 4-tensor: Inner MPO
# 3-tensor: Inner MPS, Outer MPO, Half contracted inner MPO-MPS
# 2-tensor: Outer MPS, Half contracted outer MPO-MPS, Fully contracted lattice point

# This all works assuming the MPS has form (2 x d), (d x d x 2), (2 x d)

# How to read: If we are at tensor A, what other tensor B can I contract with?
def contract_horizontal(A, B, dir):
    if A.ndim == 3:
        if B.ndim == 4: 
            if dir =='right':
                tensor = np.einsum('ijk, ibcd->bjckd', A, B)
                # Reshape to (b, j*c, k*d)
                tensor = np.reshape(tensor, (B.shape[1], A.shape[1]*B.shape[2], A.shape[2]*B.shape[3]))
            elif dir == 'left':
                tensor = np.einsum('ijk, aicd->ajckd', A, B)
                # Reshape to (a, j*c, k*d)
                tensor = np.reshape(tensor, (B.shape[0], A.shape[1]*B.shape[2], A.shape[2]*B.shape[3]))
        elif B.ndim == 3: # Used for contraction of MPO itself
            if dir == 'right' or 'left':  # Can be removed, left for readability
                tensor = np.einsum('ijk, ibc->jbkc', A, B)
                # Reshape collapses indices to (j*b, k*c)
                tensor = np.reshape(tensor, (A.shape[1]*B.shape[1], A.shape[2]*B.shape[2]))
    
    elif A.ndim == 2:
        if B.ndim == 3:
            if dir == 'right':
                tensor = np.einsum('ij, jbc->icb', A, B)
                # Reshape to (i*c, b)
                tensor = np.reshape(tensor, (A.shape[0]*B.shape[2], B.shape[1]))
            elif dir == 'left':
                tensor = np.einsum('ij, ajc->ica', A, B)
                # Reshape to (i*c, a)
                tensor = np.reshape(tensor, (A.shape[0]*B.shape[2], B.shape[0]))
        elif B.ndim == 2:
            if dir == 'right' or 'left': # Direction independent since both MPS edges are (2 x d)
                tensor = np.einsum('ij, aj->ia', A, B)
                # Reshape to (i*b)
                tensor = np.reshape(tensor, (A.shape[0]*B.shape[0]))

    elif A.ndim == 1:
        if B.ndim == 2: # Final contraction before scalar product
            if dir == 'right':
                tensor = np.einsum('i, ib->b', A, B)
            elif dir == 'left':
                tensor = np.einsum('i, ai->a', A, B)
        elif B.ndim == 1: # Inner product
            tensor = np.einsum('i, i', A, B)

    return tensor


def contract_vertical(A, B, dir):
    if A.ndim == 3:
        if B.ndim == 4:
            if dir == 'down' or 'up':
                tensor = np.einsum('ijk, abkd->iajbd', A, B) # Contracts (d x d x 2) and (3 x 3 x 2 x 2)
                # Reshape to (i*a, j*b, d)
                tensor = np.reshape(tensor, (A.shape[0]*B.shape[0], A.shape[1]*B.shape[1], B.shape[3]))
        elif B.ndim == 3:
            if dir == 'down' or 'up': # Contract (3d x 3d x 2) and (d x d x 2)
                tensor = np.einsum('ijk, abk->iajb', A, B)
                # Reshape to (i*a, j*b)
                tensor = np.reshape(tensor, (A.shape[0]*B.shape[0], A.shape[1]*B.shape[1]))

    elif A.ndim == 2:
        if B.ndim == 3:
            if dir == 'down':  # From Bra->Operator->Ket
                tensor = np.einsum('ij, abi->jab', A, B) # Contract (2 x d) and (3 x 2 x 2)
                # Reshape to (j*a, b)
                tensor = np.reshape(tensor, (A.shape[1]*B.shape[0], B.shape[1]))
            elif dir == 'up':  # From Ket->Operator->Bra
                tensor = np.einsum('ij, aic->jac', A, B)
                # Reshape to (j*a, c)
                tensor = np.reshape(tensor, (A.shape[1]*B.shape[0], B.shape[2]))
        elif B.ndim == 2:
            if dir == 'down' or 'up':
                tensor = np.einsum('ij, jb->ib', A, B) # Contract (3d x 2) and (2 x d)
                # Reshape to (i*b)
                tensor = np.reshape(tensor, (A.shape[0]*B.shape[1]))

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

# TODO: Determine if vertical/horizontal can be generalized and combined
# TODO: Use tensordot for better or standardize indices for better readability
# NOTE: In DMRG, the dimension of the final lattice position
#       will be a good marker for when to reverse direction
# NOTE: All directions of contractions have been tested and work correctly.
#       This was verified by checking if result is equal for going
#       left<->right and up<->down.


##################### INITIALIZATION MPO ######################################
# Operators
pauli_z = np.array([[1, 0],
                    [0, -1]])

pauli_x = np.array([[0, 1],
                    [1, 0]])

zero = np.zeros((2, 2))
identity = np.identity(2)
# Interaction parameter
g = 2
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

MPO_initial = matrixProductOperator(left_bound, inner, right_bound)
MPO = [MPO_initial.left_bound] + [MPO_initial.inner]*(N-2) + [MPO_initial.right_bound]
# NOTE: The MPO never gets modified, so for a possible performance boost
#       we could just use the MPO_initial. For readability by indexing over
#       each position, we will expand it anyway so that it matches the MPS.


# %%
######################### INITIALIZATION MPS ##################################
# Bond Dimension
d = 2

state =  np.zeros((1, 2*d*d))
state[0, 0] = 1

# Dimensions (2 x d)
A_1 = np.zeros((2, d))
A_1[0, 0] = 1

# Dimensions (d x d x 2)
A_i = np.reshape(state, (d, d, 2))

# Dimensions (2 x d)
A_N = A_1

MPS_initial = matrixProductState(A_1, A_i, A_N)
MPS = [MPS_initial.left_bound] + [MPS_initial.inner]*(N-2) + [MPS_initial.right_bound]

############# OLD METHOD ###############
# We initialize each wavefunction matrix as:
# A_1_ket = ( |+>    A_i_ket = ( |+X+|      A_N_ket = (|+> |->)
#             |-> )              |-x-| )

#up_ket = np.zeros((d, 1))
#up_ket[0, 0] = 1
#down_ket = np.zeros((d, 1))
#down_ket[1, 0] = 1

# # Dimensions (2 x 1 x d) - > (2 x d)
# A_1 = np.array([[up_ket],
#                 [down_ket]])
# A_1 = np.squeeze(A_1)
# # Dimensions (d x 2)
# A_1_dagger = np.array(np.matrix(A_1).H)  # Only way to do Hermitian conjugate
# A_1_dagger = np.squeeze(A_1_dagger)

# # Dimensions (2 x 1 x d x d) -> (2 x d x d)
# A_i = np.array([[np.outer(up_ket, up_ket)],  # Cannot do Hermitian conjugate of tensor
#                 [np.outer(down_ket, down_ket)]])
# A_i = np.squeeze(A_i)
# # Dimensions (1 x 2 x d x d) -> (2 x d x d)
# A_i_dagger = np.array([[np.outer(up_ket, up_ket), np.outer(down_ket, down_ket)]])
# A_i_dagger = np.squeeze(A_i_dagger)

# # Dimensions (1 x 2 x d) -> (2 x d)
# A_N = np.array([up_ket, down_ket])
# A_N = np.squeeze(A_N)
# # Dimensions (2 x 1 x d) -> (2 x d)
# A_N_dagger = np.array(np.matrix(A_N).H)
# A_N_dagger = np.squeeze(A_N_dagger)

# # Initialization of MPS elements
# MPS_ket_initial = matrixProductState(A_1, A_i, A_N)
# MPS_bra_initial = matrixProductState(A_1_dagger, A_i_dagger, A_N_dagger)
# # The inner matrices will not stay the same so we a matrix for each lattice position
# MPS_ket = [MPS_ket_initial.left_bound] + [MPS_ket_initial.inner]*(N-2) + [MPS_ket_initial.right_bound]
# MPS_bra = [MPS_bra_initial.left_bound] + [MPS_bra_initial.inner]*(N-2) + [MPS_bra_initial.right_bound]

# TODO: Verify if A_i == A_i_dagger is correct aka if squeeze is necessary/working correctly
# NOTE: May be necessary to reshape wavefunctions to D x D x 2


# %%
######################### TESTING #############################################
E_D_R = calculateExpectation(MPS, MPO, MPS, 'down', 'right')
E_D_L = calculateExpectation(MPS, MPO, MPS, 'down', 'left')
E_U_R = calculateExpectation(MPS, MPO, MPS, 'up', 'right')
E_U_L = calculateExpectation(MPS, MPO, MPS, 'up', 'left')

if (E_D_R == E_D_L == E_U_R == E_U_L):
    print("Expectation value is the same in all directions")
    print(E_D_R)


# %%
# ############## MANUAL TESTS FOR N = 3 #################################
# ####### CONTRACT HAMILTONIAN LEFT AND RIGHT ########
# temp = contract_horizontal(MPO[0], MPO[1], 'right')
# H_right = contract_horizontal(temp, MPO[2], 'right')
# temp = contract_horizontal(MPO[2], MPO[1], 'left')
# H_left = contract_horizontal(temp, MPO[0], 'left')
# if (H_right.all() == H_left.all()):
#     print("Hamiltonian contracts correctly")

# ######## EXPECTATION VALUE DOWN, RIGHT ##########
# # Left lattice bra->H->ket
# temp = contract_vertical(MPS[0], MPO[0], 'down')
# left = contract_vertical(temp, MPS[0], 'down')

# # Inner lattice bra->H->ket
# temp = contract_vertical(MPS[1], MPO[1], 'down')
# inner = contract_vertical(temp, MPS[1], 'down')

# # Right lattice bra->H->ket
# temp = contract_vertical(MPS[2], MPO[2], 'down')
# right = contract_vertical(temp, MPS[2], 'down')

# temp = contract_horizontal(left, inner, 'right')
# E_down_right = contract_horizontal(temp, right, 'right')

# ###### EXPECTATION VALUE UP, RIGHT ###########
# # Left lattice ket->H->bra
# temp = contract_vertical(MPS[0], MPO[0], 'up')
# left = contract_vertical(temp, MPS[0], 'up')

# # Inner lattice ket->H->bra
# temp = contract_vertical(MPS[1], MPO[1], 'up')
# inner = contract_vertical(temp, MPS[1], 'up')

# # Right lattice ket->H->bra
# temp = contract_vertical(MPS[2], MPO[2], 'up')
# right = contract_vertical(temp, MPS[2], 'up')

# temp = contract_horizontal(left, inner, 'right')
# E_up_right = contract_horizontal(temp, right, 'right')

# ###### EXPECTATION VALUE UP, LEFT ###########
# # Right lattice bra->H->ket
# temp = contract_vertical(MPS[2], MPO[2], 'up')
# right = contract_vertical(temp, MPS[2], 'up')

# # Inner lattice bra->H->ket
# temp = contract_vertical(MPS[1], MPO[1], 'up')
# inner = contract_vertical(temp, MPS[1], 'up')

# # Left lattice bra->H->ket
# temp = contract_vertical(MPS[0], MPO[0], 'up')
# left = contract_vertical(temp, MPS[0], 'up')

# temp = contract_horizontal(right, inner, 'left')
# E_up_left = contract_horizontal(temp, left, 'left')

# ###### EXPECTATION VALUE DOWN, LEFT ###########
# # Right lattice bra->H->ket
# temp = contract_vertical(MPS[2], MPO[2], 'down')
# right = contract_vertical(temp, MPS[2], 'down')

# # Inner lattice bra->H->ket
# temp = contract_vertical(MPS[1], MPO[1], 'down')
# inner = contract_vertical(temp, MPS[1], 'down')

# # Left lattice bra->H->ket
# temp = contract_vertical(MPS[0], MPO[0], 'down')
# left = contract_vertical(temp, MPS[0], 'down')

# temp = contract_horizontal(right, inner, 'left')
# E_down_left = contract_horizontal(temp, left, 'left')

# ######## EXPECTATION VALUE ZIG-ZAG ##########
# # Left lattice bra->H->ket
# temp = contract_vertical(MPS[0], MPO[0], 'down')
# left = contract_vertical(temp, MPS[0], 'down')

# # Inner lattice ket->H->bra
# temp = contract_vertical(MPS[1], MPO[1], 'up')
# inner = contract_vertical(temp, MPS[1], 'up')

# # Right lattice bra->H->ket
# temp = contract_vertical(MPS[2], MPO[2], 'down')
# right = contract_vertical(temp, MPS[2], 'down')

# temp = contract_horizontal(left, inner, 'right')
# E_zig_zag = contract_horizontal(temp, right, 'right')

# if (E_down_right.all() == E_down_left.all() == E_up_right.all() == E_up_left.all() == E_zig_zag.all()):
#      print("Expectation value contracts in all directions correctly")

