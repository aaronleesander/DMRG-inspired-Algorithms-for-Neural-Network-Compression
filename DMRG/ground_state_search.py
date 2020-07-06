import math
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse.linalg
import scipy.sparse as sparse

import contractions as con
import initializations as init
import metrics


def contract_L(bra, MPO, ket, site):
    """ Contracts all tensors to the left of a given site when updating
        a site during ground state search

    Args:
        bra: MPS used as the bra
        MPO: List of tensors
        ket: MPS used as the ket
        site: Site to be updated

    Returns:
        L: Tensor with dimensions (bondW, braBondDim, ketBondDim)
    """
    for i in range(0, site):
        if i == site-1:  # We need to keep the extra leg from the bra
            if site == 1:  # L tensor is left bound only
                # Contracts to (bondW, a_1, a_1')
                pos = np.einsum('ij, ali, lm->ajm', bra[i], MPO[i], ket[i])
                # (a) bond to W, (j) leftover leg from bra, (m) leftover leg from ket
                L = pos
            else:
                pos = np.einsum('ijk, abkn, lmn->ialbjm', bra[i], MPO[i], ket[i])
                # (ial) will connect to previous L, (b) bond to W, (j) leftover bra leg, (m) leftover ket leg
                pos = np.reshape(pos, (pos.shape[0]*pos.shape[1]*pos.shape[2],
                                       pos.shape[3], pos.shape[4],
                                       pos.shape[5]))
                L = np.einsum('i, ibcd->bcd', L, pos)
        else:  # Normal tensor contraction at given position
            pos = con.contract_vertical(bra[i], MPO[i], dir='down')
            pos = con.contract_vertical(pos, ket[i], dir='down')
            if i == 0:  # Initialize L
                L = pos
            else:  # Add pos onto previous L
                L = con.contract_horizontal(L, pos, dir='right')
    return L


def contract_R(bra, MPO, ket, site):
    """ Contracts all tensors to the right of a given site when updating
        a site during ground state search

    Args:
        bra: MPS used as the bra
        MPO: List of tensors
        ket: MPS used as the ket
        site: Site to be updated

    Returns:
        R: Tensor with dimensions (bondW, braBondDim, ketBondDim)
    """
    for i in range(len(bra)-1, site, -1):
        if i == site+1:  # We need to keep the extra leg from the bra
            if i == len(bra)-1:  # R tensor is the right bound only
                # Contracts to (bondW, a_1, a_1)
                pos = np.einsum('ij, ali, lm->ajm', bra[i], MPO[i], ket[i])
                # (a) bond to W, (j) leftover leg from bra, (m) leftover leg from ket
                R = pos
            else:
                pos = np.einsum('ijk, abkn, lmn->ailjbm', bra[i], MPO[i], ket[i])
                # (a) bond to W, (i) leftover leg from bra, (l) leftover leg from ket, (jbm) will connect to previous R
                pos = np.reshape(pos, (pos.shape[0],
                                       pos.shape[1],
                                       pos.shape[2],
                                       pos.shape[3]*pos.shape[4]*pos.shape[5]))
                R = np.einsum('ijkl, l->ijk', pos, R)
        else:  # Normal tensor contraction at given position
            pos = con.contract_vertical(bra[i], MPO[i], dir='down')
            pos = con.contract_vertical(pos, ket[i], dir='down')
            if i == len(bra)-1:  # Initialize R
                R = pos
            else:  # Add pos onto previous R
                R = con.contract_horizontal(pos, R, dir='right')
    return R


def create_Hamiltonian(bra, MPO, ket, site):
    """ Contracts MPS-MPO-MPS expectation value with missing site
        to give Hamiltonian derivative

    Args:
        bra: MPS used as the bra
        MPO: List of tensors
        ket: MPS used as the ket (indices marked with ')
        site: Site to be updated

    Returns:
        H: Rank-2 tensor with shape ( (braIndices) x (ketIndices) )
    """
    # Create L, R, and W for system
    if site != 0:
        L = contract_L(bra, MPO, ket, site)
    if site != len(bra)-1:
        R = contract_R(bra, MPO, ket, site)
    W = MPO[site]

    if site == 0:
        # (bondW, sigma_l', sigma_l) x (bondW, a_l, a_l')
        # -> (sigma_l, a_l, sigma_l', a_l')
        H = np.einsum('ijk, ibc->kbjc', W, R)
        H = np.reshape(H, (H.shape[0]*H.shape[1], H.shape[2]*H.shape[3]))
    elif site == len(bra)-1:
        # (bondW, a_(l-1), a_(l-1)') x (bondW, sigma_l', sigma_l)
        # -> (sigma_l, a_(l-1), sigma_l', a_(l-1)')
        H = np.einsum('ijk, ibc->cjbk', L, W)
        H = np.reshape(H, (H.shape[0]*H.shape[1], H.shape[2]*H.shape[3]))
    else:
        # (bondW, a_(l-1), a_(l-1)') x (bondL, bondR, sigma_l', sigma_l) x (bondW, a_l, a_l')
        # Shape (sigma_l, a_(l-1), a_l, sigma_l', a_(l-1)', a_l')
        H = np.einsum('ijk, ilcd, lmn->djmckn', L, W, R)
        H = np.reshape(H, (H.shape[0]*H.shape[1]*H.shape[2],
                           H.shape[3]*H.shape[4]*H.shape[5]))
    return H


def update_site(bra, MPO, ket, site, dir):
    """ Updates a given site of an MPS during the ground state search

    Args:
        bra: MPS used as the bra
        MPO: List of tensors
        ket: MPS used as the ket (indices marked with ')
        site: Site to be updated
        dir: Direction of sweep

    Returns:
        updated_site: Updated tensor at current site
        next_site_M: M tensor to replaced neighboring site
                     either directly left or right of current site based on
                     direction of sweep
    """
    # Hamiltonian without site to be updated
    H = create_Hamiltonian(bra, MPO, ket, site)

    # Initial eigenstate
    if bra[site].ndim == 2:
        v0 = np.reshape(bra[site], (bra[site].shape[0]*bra[site].shape[1]))
    elif bra[site].ndim == 3:
        # Needs shape v = (sigma, a_l-1, a_l) for eigenvalue problem
        permuted = np.transpose(bra[site], (2, 0, 1))
        v0 = np.reshape(permuted, (permuted.shape[0]
                                   * permuted.shape[1]
                                   * permuted.shape[2]))

    # Solve eigenvalue problem for lowest value eigenvalue and its eigenstate
    E, V = sparse.linalg.eigsh(H, k=1, v0=v0, which='SA', tol=1E-8)

    # Reshape eigenstate to correct shape
    if bra[site].ndim == 2:
        updated_M = np.reshape(V, (bra[site].shape))
    else:
        updated_M = np.reshape(V, (permuted.shape))
        updated_M = np.transpose(updated_M, (1, 2, 0))

    if dir == 'right':
        # Inner tensor needs to be reshaped
        if updated_M.ndim == 3:
            reshaped_M = np.transpose(updated_M, (0, 2, 1))  # Move left bond and physical dimension together
            reshaped_M = np.reshape(reshaped_M, (reshaped_M.shape[0]*reshaped_M.shape[1], reshaped_M.shape[2]))
            U, S_vector, V = np.linalg.svd(reshaped_M, full_matrices=False)
            A_tensor = np.reshape(U, (bra[site].shape[0], bra[site].shape[2], U.shape[1]))
            A_tensor = np.transpose(A_tensor, (0, 2, 1))
        else:
            U, S_vector, V = np.linalg.svd(updated_M, full_matrices=False)
            A_tensor = U

        lambda_tensor = np.diag(S_vector)

        if site == len(bra)-2:  # Multiplies with matrix on right bound (dim 2 x d)
            next_site_M = np.einsum('ij, jb, lb->li', lambda_tensor, V, bra[site+1])
        else:
            next_site_M = np.einsum('ij, jb, bmn->imn', lambda_tensor, V, bra[site+1])

        updated_site = A_tensor

    elif dir == 'left':
        if updated_M.ndim == 3:
            reshaped_M = np.reshape(updated_M, (updated_M.shape[0], updated_M.shape[1]*updated_M.shape[2]))
            U, S_vector, V = np.linalg.svd(reshaped_M, full_matrices=False)
            B_tensor = np.reshape(V, (V.shape[0], bra[site].shape[1], bra[site].shape[2]))
        else:
            U, S_vector, V = np.linalg.svd(updated_M.T, full_matrices=False)  # Transpose to have shape (2 x d)
            B_tensor = V.T

        lambda_tensor = np.diag(S_vector)

        if site == 1:
            next_site_M = np.einsum('ij, jb, bm->im', bra[site-1], U, lambda_tensor)
        else:
            next_site_M = np.einsum('ijk, jb, bm->imk', bra[site-1], U, lambda_tensor)

        updated_site = B_tensor

    return updated_site, next_site_M


def ground_state_search(MPO, threshold, plot=0):
    """ Solves the eigenvalue equation HV = EV until we converge to an
        energy value.

    Args:
        MPO: List of tensors representing an operator
        threshold: Difference between sweeps under which a solution is found
        plot: Whether or not to plot the eigenvalues (0 off, 1 on)

    Returns:
        eigenvalues: Ground state energy for each bond dimension
        eigenstates: Eigenstate MPS at each bond dimension
    """

    # Initial state with max bond dimension 1
    max_bond_dim = 1
    MPS = init.initialize_random_normed_state_MPS(num_sites=len(MPO),
                                                  bond_dim=max_bond_dim,
                                                  phys_dim=MPO[0].shape[2])

    # Initial energy value
    E = []
    E.append(con.calculate_expectation(MPS, MPO, MPS))
    print("Initial Energy:", E[-1])

    # Initialize result arrays
    eigenvalues = []
    eigenstates = []

    # Initialize stopping value
    last_bond_dim_E = E[-1]
    while True:
        # Left->right sweep
        for site in range(0, len(MPS)-1):
            MPS[site], MPS[site+1] = update_site(MPS, MPO, MPS,
                                                 site=site,
                                                 dir='right')

        # Right->left sweep
        for site in range(len(MPS)-1, 0, -1):
            MPS[site], MPS[site-1] = update_site(MPS, MPO, MPS,
                                                 site=site,
                                                 dir='left')

        # Update energy value after every full sweep
        E.append(con.calculate_expectation(MPS, MPO, MPS)
                 / metrics.scalar_product(MPS, MPS))
        print("Energy:", E[-1], "BondDim:", max_bond_dim)

        # Check if sweeps are still working
        if np.abs(E[-2]-E[-1]) < threshold:
            # Save the result for each bond dimension
            eigenvalues.append(E[-1])
            eigenstates.append(MPS[:])

            # Check if the increase in bond dimension did enough
            if np.abs(last_bond_dim_E-E[-1]) < threshold:
                break
            last_bond_dim_E = E[-1]

            # Update each tensor by increasing bond dimension
            for i, tensor in enumerate(MPS):
                if tensor.ndim == 2:
                    new_tensor = np.zeros((tensor.shape[0], tensor.shape[1]+1))
                    new_tensor[:tensor.shape[0], :tensor.shape[1]] = tensor
                    MPS[i] = new_tensor

                elif tensor.ndim == 3:
                    new_tensor = np.zeros((tensor.shape[0]+1, tensor.shape[1]+1, tensor.shape[2]))
                    new_tensor[:tensor.shape[0], :tensor.shape[1], :tensor.shape[2]] = tensor
                    MPS[i] = new_tensor
            max_bond_dim = MPS[math.ceil(len(MPS)/2)].shape[0]

    if plot == 1:
        plt.figure()
        plt.title("Energy Eigenvalue vs. Bond Dimension")
        plt.xlabel("Max Bond Dimension")
        plt.ylabel("Energy")

        plt.plot(range(1, len(eigenvalues)+1), eigenvalues)

    return eigenvalues, eigenstates, max_bond_dim
