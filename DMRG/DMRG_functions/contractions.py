import numpy as np


def contract_horizontal(A, B, dir):
    """ Hardcoded contraction of tensors in an MPS/MPO/MPS network
        based on the number of legs of the two tensors and their horizontal
        directional relationship A->B
    Args:
        A: First Tensor
        B: Second Tensor
        dir: Horizontal direction A->B in the MPS/MPO/MPS network
            ('left' or 'right')

    Returns:
        tensor: Contracted tensor C = AB
    """
    if A.ndim == 3:
        if B.ndim == 4:
            if dir == 'right':
                tensor = np.einsum('ijk, ibcd->bjckd', A, B)
                # Reshape to (b, j*c, k*d)
                tensor = np.reshape(tensor, (B.shape[1], A.shape[1]*B.shape[2],
                                             A.shape[2]*B.shape[3]))
            elif dir == 'left':
                tensor = np.einsum('ijk, aicd->ajckd', A, B)
                # Reshape to (a, j*c, k*d)
                tensor = np.reshape(tensor, (B.shape[0], A.shape[1]*B.shape[2],
                                             A.shape[2]*B.shape[3]))
        elif B.ndim == 3:  # Used for contraction of MPO with itself
            if dir == 'right' or 'left':  # Left for readability
                tensor = np.einsum('ijk, ibc->jbkc', A, B)
                # Reshape collapses indices to (j*b, k*c)
                tensor = np.reshape(tensor, (A.shape[1]*B.shape[1],
                                             A.shape[2]*B.shape[2]))

    elif A.ndim == 2:
        if B.ndim == 3:
            if dir == 'right':
                tensor = np.einsum('ij, jbc->icb', A, B)
                # Reshape to (i*c, b)
                tensor = np.reshape(tensor, (A.shape[0]*B.shape[2],
                                             B.shape[1]))
            elif dir == 'left':
                tensor = np.einsum('ij, ajc->ica', A, B)
                # Reshape to (i*c, a)
                tensor = np.reshape(tensor, (A.shape[0]*B.shape[2],
                                             B.shape[0]))
        elif B.ndim == 2:
            if dir == 'right' or 'left':  # Direction independent since both MPS edges are (2 x d)
                tensor = np.einsum('ij, aj->ia', A, B)
                # Reshape to (i*a)
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
    """ Hardcoded contraction of tensors in an MPS/MPO/MPS network
        based on the number of legs of the two tensors and their vertical
        directional relationship A->B
    Args:
        A: First Tensor
        B: Second Tensor
        dir: Vertical direction A->B in the MPS/MPO/MPS network
             ('up' or 'down')

    Returns:
        tensor: Contracted tensor C = AB
    """
    if A.ndim == 3:
        if B.ndim == 4:
            if dir == 'down':
                tensor = np.einsum('ijk, abck->iajbc', A, B)  # Contracts (d x d x 2) and (3 x 3 x 2 x 2)
                # Reshape to (i*a, j*b, c)
                tensor = np.reshape(tensor, (A.shape[0]*B.shape[0],
                                             A.shape[1]*B.shape[1],
                                             B.shape[3]))
            elif dir == 'up':
                tensor = np.einsum('ijk, abkd->iajbd', A, B)  # Contracts (d x d x 2) and (3 x 3 x 2 x 2)
                # Reshape to (i*a, j*b, d)
                tensor = np.reshape(tensor, (A.shape[0]*B.shape[0],
                                             A.shape[1]*B.shape[1],
                                             B.shape[3]))
        elif B.ndim == 3:
            if dir == 'down' or 'up':  # Contract (3d x 3d x 2) and (d x d x 2)
                tensor = np.einsum('ijk, abk->iajb', A, B)
                # Reshape to (i*a, j*b)
                tensor = np.reshape(tensor, (A.shape[0]*B.shape[0],
                                             A.shape[1]*B.shape[1]))

    elif A.ndim == 2:
        if B.ndim == 3:
            if dir == 'down':  # From Bra->Operator->Ket
                tensor = np.einsum('ij, abi->jab', A, B)  # Contract (2 x d) and (3 x 2 x 2)
                # Reshape to (j*a, b)
                tensor = np.reshape(tensor, (A.shape[1]*B.shape[0],
                                             B.shape[1]))
            elif dir == 'up':  # From Ket->Operator->Bra
                tensor = np.einsum('ij, aic->jac', A, B)
                # Reshape to (j*a, c)
                tensor = np.reshape(tensor, (A.shape[1]*B.shape[0],
                                             B.shape[2]))
        elif B.ndim == 2:
            if dir == 'down' or 'up':
                tensor = np.einsum('ij, jb->ib', A, B)  # Contract (3d x 2) and (2 x d)
                # Reshape to (i*b)
                tensor = np.reshape(tensor, (A.shape[0]*B.shape[1]))

    return tensor


def calculate_expectation(MPS_bra, MPO, MPS_ket,
                          vert_dir='down', horiz_dir='right'):

    """ Hardcoded contraction of tensors in an MPS/MPO/MPS network
        based on the number of legs of the two tensors and their horizontal
        directional relationship A->B

    Args:
        MPS_bra: List of MPS tensors used as the bra state
        MPO: List of MPO tensors
        MPS_ket: List of MPS tensors used as the ket state
        vert_dir: Specifies direction of contracting MPS/MPO/MPS
                'Down': Bra -> MPO -> Ket
                'Up': Ket -> MPO -> Bra
        horiz_dir: Specifies direction of contracting after vertical contraction
                'Right': Left Bound -> Inner -> Right Bound
                'Left': Right Bound -> Inner -> Left Bound

        Default direction is 'down' and 'right'
    Returns:
        E: Operation <A|MPO|B>
    """
    tensor = [None]*len(MPO)

    for i in range(0, len(MPO)):
        if vert_dir == 'down':
            first_contraction = contract_vertical(MPS_bra[i], MPO[i], vert_dir)
            tensor[i] = contract_vertical(first_contraction, MPS_ket[i],
                                          vert_dir)
        if vert_dir == 'up':
            first_contraction = contract_vertical(MPS_ket[i], MPO[i], vert_dir)
            tensor[i] = contract_vertical(first_contraction, MPS_bra[i],
                                          vert_dir)

    if horiz_dir == 'right':
        E = tensor[0]
        for i in range(1, len(tensor)):
            E = contract_horizontal(E, tensor[i], horiz_dir)
    if horiz_dir == 'left':
        E = tensor[-1]
        for i in range(len(tensor)-2, -1, -1):
            E = contract_horizontal(E, tensor[i], horiz_dir)

    return E


def check_expectation_value_contractions(MPS, MPO):
    """ Tests if the result for the expectation value of an MPS and MPO
        is the same for all directions of contraction. If this is not true,
        there may be a bug somewhere.

    Args:
        MPS: list of tensors
        MPO: list of tensors

    Returns:
        prints expectation value based on direction and checks if they are
        all the same
    """
    E_D_R = calculate_expectation(MPS, MPO, MPS, 'down', 'right')
    E_D_L = calculate_expectation(MPS, MPO, MPS, 'down', 'left')
    E_U_R = calculate_expectation(MPS, MPO, MPS, 'up', 'right')
    E_U_L = calculate_expectation(MPS, MPO, MPS, 'up', 'left')

    # Rounding necessary since sometimes the values are slightly off
    # Most likely due to rounding errors in computation
    if (np.round(E_D_R, 5) == np.round(E_D_L, 5)
            == np.round(E_U_R, 5)
            == np.round(E_U_L, 5)):
        print("Expectation value is the same in all directions")
    print(E_D_R, E_D_L, E_U_R, E_U_L)
