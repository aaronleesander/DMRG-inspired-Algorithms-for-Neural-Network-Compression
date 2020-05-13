###################### FUNCTIONS FOR MPO ######################################

import numpy as np

# Used to contract tensors horizontally from A->B
# All 2 dimensional tensors have dimensions (ij) or (ab)
# All 3 dimensional tensors have dimensions (ijk) or (abc)
# All 4 dimensional tensors have dimensions (ijkl) or (abcd)

# Examples:
# 4-tensor: Inner MPO
# 3-tensor: Inner MPS, Outer MPO, Half contracted inner MPO-MPS
# 2-tensor: Outer MPS, Half contracted outer MPO-MPS, Fully contracted lattice point

# This all works assuming the MPS has form (2 x d), (d x d x 2), (2 x d)
# Only valid for full system with MPS, MPO, MPS.
# How to read: If we are at tensor A, what other tensor B can I contract with?


def contract_horizontal(A, B, dir):
    if A.ndim == 3:
        if B.ndim == 4:
            if dir == 'right':
                tensor = np.einsum('ijk, ibcd->bjckd', A, B)
                # Reshape to (b, j*c, k*d)
                tensor = np.reshape(tensor, (B.shape[1], A.shape[1]*B.shape[2], A.shape[2]*B.shape[3]))
            elif dir == 'left':
                tensor = np.einsum('ijk, aicd->ajckd', A, B)
                # Reshape to (a, j*c, k*d)
                tensor = np.reshape(tensor, (B.shape[0], A.shape[1]*B.shape[2], A.shape[2]*B.shape[3]))
        elif B.ndim == 3:  # Used for contraction of MPO itself
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
            if dir == 'right' or 'left':  # Direction independent since both MPS edges are (2 x d)
                tensor = np.einsum('ij, aj->ia', A, B)
                # Reshape to (i*b)
                tensor = np.reshape(tensor, (A.shape[0]*B.shape[0]))
        elif B.ndim == 1:
            if dir == 'right':
                tensor = np.einsum('ij, j->i', A, B)

    elif A.ndim == 1:
        if B.ndim == 2:  # Final contraction before scalar product
            if dir == 'right':
                tensor = np.einsum('i, ib->b', A, B)
            elif dir == 'left':
                tensor = np.einsum('i, ai->a', A, B)
        elif B.ndim == 1:  # Inner product
            tensor = np.einsum('i, i', A, B)

    return tensor


def contract_vertical(A, B, dir):
    if A.ndim == 3:
        if B.ndim == 4:
            if dir == 'down':
                tensor = np.einsum('ijk, abck->iajbc', A, B)  # Contracts (d x d x 2) and (3 x 3 x 2 x 2)
                # Reshape to (i*a, j*b, c)
                tensor = np.reshape(tensor, (A.shape[0]*B.shape[0], A.shape[1]*B.shape[1], B.shape[3]))
            elif dir == 'up':
                tensor = np.einsum('ijk, abkd->iajbd', A, B)  # Contracts (d x d x 2) and (3 x 3 x 2 x 2)
                # Reshape to (i*a, j*b, d)
                tensor = np.reshape(tensor, (A.shape[0]*B.shape[0], A.shape[1]*B.shape[1], B.shape[3]))
        elif B.ndim == 3:
            if dir == 'down' or 'up':  # Contract (3d x 3d x 2) and (d x d x 2)
                tensor = np.einsum('ijk, abk->iajb', A, B)
                # Reshape to (i*a, j*b)
                tensor = np.reshape(tensor, (A.shape[0]*B.shape[0], A.shape[1]*B.shape[1]))

    elif A.ndim == 2:
        if B.ndim == 3:
            if dir == 'down':  # From Bra->Operator->Ket
                tensor = np.einsum('ij, abi->jab', A, B)  # Contract (2 x d) and (3 x 2 x 2)
                # Reshape to (j*a, b)
                tensor = np.reshape(tensor, (A.shape[1]*B.shape[0], B.shape[1]))
            elif dir == 'up':  # From Ket->Operator->Bra
                tensor = np.einsum('ij, aic->jac', A, B)
                # Reshape to (j*a, c)
                tensor = np.reshape(tensor, (A.shape[1]*B.shape[0], B.shape[2]))
        elif B.ndim == 2:
            if dir == 'down' or 'up':
                tensor = np.einsum('ij, jb->ib', A, B)  # Contract (3d x 2) and (2 x d)
                # Reshape to (i*b)
                tensor = np.reshape(tensor, (A.shape[0]*B.shape[1]))

    return tensor


## Calculates the expectation value by vertical contraction then horizontal contraction ###
def calculate_expectation(MPS_bra, MPO, MPS_ket, vert_dir, horiz_dir):
    # Initialize list of tensors
    tensor = [None]*len(MPO)

    # Contract <MPS|MPO|MPS> at each lattice position
    # Down: Bra -> MPO -> Ket
    # Up: Ket -> MPO -> Bra
    for i in range(0, len(MPO)):
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
# TODO: Cut singular values under a threshold
# NOTE: All directions of contractions have been tested and work correctly.
#       This was verified by checking if result is equal for going
#       left<->right and up<->down.
