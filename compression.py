### COMPRESSION SWEEPS ###

import numpy as np
from metrics import *
from canonical_forms import *

### Bra and Ket are mixed AAMBB-type tensor networks ####
### Tensor indices are the same as in DMRG by Schollwöck ###
# TODO: Merge with normalization or create subfunctions


def update_site(bra, ket, site, dir):
    ### Bra is compressed state, ket is raw state
    ### L TENSOR WITH SHAPE (bondDim_compressed, bondDim_raw) ###
    for i in range(0, site):
        if i == 0:  # Left bound
            if i == site-1:  # L tensor is the left bound only
                L = np.einsum('ij, ib->jb', bra[i], ket[i])
            else:
                pos = np.einsum('ij, ib->jb', bra[i], ket[i])
                pos = np.reshape(pos, (bra[i].shape[1]*ket[i].shape[1]))
                L = pos
        elif i == site-1:  # We need to keep the leg of the bra that connects to the missing tensor
            pos = np.einsum('ijk, abk->iajb', bra[i], ket[i])
            # (ia) will connect to previous L, (j) will be left bond of final tensor, (b) will connect to M
            # -> (Left bond, final tensor leg, connects to M)
            pos = np.reshape(pos, (bra[i].shape[0]*ket[i].shape[0], bra[i].shape[1], ket[i].shape[1]))

            # L has shape (final tensor left bond, connects to M left bond)
            L = np.einsum('i, ibc->bc', L, pos)
        else:  # Inner sites
            pos = np.einsum('ijk, abk->iajb', bra[i], ket[i])
            pos = np.reshape(pos, (bra[i].shape[0]*ket[i].shape[0], bra[i].shape[1]*ket[i].shape[1]))
            L = np.einsum('i, ij->j', L, pos)

    ### M TENSOR AT SITE, UNCHANGED ###
    M = ket[site]

    ### R TENSOR WITH SHAPE (bondDim_compressed, bondDim_raw) ###
    for i in range(len(bra)-1, site, -1):
        if i == len(bra)-1:
            if i == site+1:  # R tensor is the right bound only
                R = np.einsum('ij, ib->jb', bra[i], ket[i])
            else:
                pos = np.einsum('ij, ib->jb', bra[i], ket[i])
                pos = np.reshape(pos, (bra[i].shape[1]*ket[i].shape[1]))
                R = pos

        elif i == site+1:  # We need to keep the leg of the bra that connects to the missing tensor
            pos = np.einsum('ijk, abk->iajb', bra[i], ket[i])
            # (i) will be right bond of final tensor, (a) will connect to M, (jb) will connect to previous R
            # -> (Left bond, right bond, final tensor left leg)
            pos = np.reshape(pos, (bra[i].shape[0], ket[i].shape[0], bra[i].shape[1]*ket[i].shape[1]))
            # R has shape (Connects to M right bound, final tensor right bound)

            R = np.einsum('ijk, k->ij', pos, R)

        else:  # Inner sites
            pos = np.einsum('ijk, abk->iajb', bra[i], ket[i])
            pos = np.reshape(pos, (bra[i].shape[0]*ket[i].shape[0], bra[i].shape[1]*ket[i].shape[1]))
            R = np.einsum('ij, j->i', pos, R)

    ### CONTRACT M' = LMR ###
    if site == 0:
        updated_M = np.einsum('ij, aj->ia', M, R)
    elif site == len(bra)-1:
        updated_M = np.einsum('ij, aj->ai', L, M)
    else:
        updated_M = np.einsum('ij, jbc, ab->iac', L, M, R)

    ### UPDATING SITES ###

    # For a left->right sweep, similar to left normalization
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

    # For a right->left sweep, similar to right normalization
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


def full_sweep(compressed_state, raw_state, threshold):
    # We initialize the compressed state so that it can be updated.
    # Compressed state must start right normalized for a left sweep (and vice versa)
    A_tensors, lambda_tensors = left_normalize(compressed_state)
    gamma_tensors, _ = vidal_notation(A_tensors, lambda_tensors, normalization='left')

    # Initialize site canonical form at first site
    mixed = site_canonical(gamma_tensors, lambda_tensors, site=0)

    # Initialize accuracy metrics
    dist = []  # Frobenius norm
    sim = []   # Cosine similarity (Scalar product)
    dist.append(overlap(mixed, raw_state))
    sim.append(scalar_product(mixed, raw_state))
    # We sweep left to right and then back right to left across the mixed state
    while True:
        # Left->right sweep
        for site in range(0, len(raw_state)-1):
            mixed[site], mixed[site+1] = update_site(mixed, raw_state, site=site, dir='right')

        # Right->left sweep
        for site in range(len(raw_state)-1, 0, -1):
            mixed[site], mixed[site-1] = update_site(mixed, raw_state, site=site, dir='left')

        # Metrics are updated after each full sweep
        dist.append(overlap(mixed, raw_state))
        sim.append(scalar_product(mixed, raw_state))
        if np.abs(sim[-2]-sim[-1]) < threshold:
            break

    return mixed, dist, sim
