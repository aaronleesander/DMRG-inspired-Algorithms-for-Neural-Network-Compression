import numpy as np
import activation_functions as act
import metrics
import initializations as init


def contract_L(bra, ket, site):
    """ Contracts all tensors to the left of a given site when updating
        a site during compression sweeps

    Args:
        bra: MPS used as the bra, compressed state
        ket: MPS used as the ket, raw state
        site: Site to be updated

    Returns:
        L: Tensor with dimensions (bond_dim_compressed, bond_dim_raw)
    """
    for i in range(0, site):
        if i == 0:
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
            pos = np.reshape(pos, (bra[i].shape[0]*ket[i].shape[0],
                                   bra[i].shape[1],
                                   ket[i].shape[1]))

            # L has shape (final tensor left bond, connects to M left bond)
            L = np.einsum('i, ibc->bc', L, pos)
        else:  # Inner sites
            pos = np.einsum('ijk, abk->iajb', bra[i], ket[i])
            pos = np.reshape(pos, (bra[i].shape[0]*ket[i].shape[0],
                                   bra[i].shape[1]*ket[i].shape[1]))
            L = np.einsum('i, ij->j', L, pos)
    return L


def contract_R(bra, ket, site):
    """ Contracts all tensors to the right of a given site when updating
        a site during compression sweeps

    Args:
        bra: MPS used as the bra, compressed state
        ket: MPS used as the ket, raw state
        site: Site to be updated

    Returns:
        R: Tensor with dimensions (bond_dim_compressed, bond_dim_raw)
    """
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
            pos = np.reshape(pos, (bra[i].shape[0], ket[i].shape[0],
                                   bra[i].shape[1]*ket[i].shape[1]))
            # R has shape (Connects to M right bound, final tensor right bound)

            R = np.einsum('ijk, k->ij', pos, R)

        else:  # Inner sites
            pos = np.einsum('ijk, abk->iajb', bra[i], ket[i])
            pos = np.reshape(pos, (bra[i].shape[0]*ket[i].shape[0],
                                   bra[i].shape[1]*ket[i].shape[1]))
            R = np.einsum('ij, j->i', pos, R)
    return R


def contract_environment(bra, ket, site):
    if site != 0:
        L = contract_L(bra, ket, site)
    if site != len(bra)-1:
        R = contract_R(bra, ket, site)
    M = ket[site]

    if site == 0:
        M = np.einsum('ij, aj->ia', M, R)
    elif site == len(bra)-1:
        M = np.einsum('ij, aj->ai', L, M)
    else:
        M = np.einsum('ij, jbc, ab->iac', L, M, R)

    return M


def gradient_descent(bra, site, grad, activation_function, learning_rate=1e-1):
    M = bra[site]
    if M.ndim == 3:
        vectorized_M = np.reshape(M, (M.shape[0]*M.shape[1]*M.shape[2]))
    elif M.ndim == 2:
        vectorized_M = np.reshape(M, (M.shape[0]*M.shape[1]))

    df_dM = vectorized_M[:]

    if activation_function == 'linear':
        df_dM[:] = 1
    elif activation_function == 'ReLU':  # XXX
        df_dM[df_dM > 0] = 1
        df_dM[df_dM <= 0] = 0
    elif activation_function == 'arctan':
        df_dM = 1/(vectorized_M**2 + 1)
    elif activation_function == 'tanh':
        df_dM = 1-np.tanh(vectorized_M)**2
    elif activation_function == 'arcsinh':
        df_dM = 1/np.sqrt(vectorized_M**2 + 1)
    elif activation_function == 'sigmoid':
        df_dM = act.sigmoid(vectorized_M) - act.sigmoid(vectorized_M)**2
    elif activation_function == 'softplus':
        df_dM = 1/(1 + np.exp(-vectorized_M))
    elif activation_function == 'SiLU':
        df_dM = (1 + np.exp(-vectorized_M) + vectorized_M * np.exp(-vectorized_M)) / (1 + np.exp(vectorized_M))**2
    elif activation_function == 'sinusoid':
        df_dM = np.cos(df_dM)

    grad = grad * df_dM
    updated_M = vectorized_M - learning_rate*grad
    updated_M = np.reshape(updated_M, (M.shape))

    # if activation_function == 'linear':
    #     updated_M = updated_A
    # elif activation_function == 'ReLU':  # XXX
    #     df_dM[df_dM > 0] = 1
    #     df_dM[df_dM <= 0] = 0
    # elif activation_function == 'arctan':
    #     df_dM = act.arctan(vectorized_M)
    # elif activation_function == 'tanh':
    #     df_dM = act.tanh(vectorized_M)
    # elif activation_function == 'arcsinh':
    #     df_dM = act.arcsinh(vectorized_M)
    # elif activation_function == 'sigmoid':
    #     df_dM = act.sigmoid(vectorized_M)
    # elif activation_function == 'softplus':
    #     df_dM = act.softplus(vectorized_M)
    # elif activation_function == 'SiLU':
    #     df_dM = act.SiLU(vectorized_M)
    # elif activation_function == 'sinusoid':
    #     updated_M = np.sin(updated_A)

    return updated_M


def update_site(bra, ket, activation_function, site, dir):
    """ Updates a given site of an MPS during the compression sweep

    Args:
        bra: MPS used as the bra, compressed state
        ket: MPS used as the ket, raw state
        dir: Current direction of sweep ('left' or 'right)

    Returns:
        updated_site: Updated tensor at current site
        next_site_M: M tensor to replaced neighboring site
                   either directly left or right of current site based on
                   direction of sweep
    """

    M1 = contract_environment(bra, bra, site)
    M2 = contract_environment(bra, ket, site)
    grad = M1 - M2

    if grad.ndim == 3:
        grad = np.reshape(grad, (grad.shape[0]*grad.shape[1]*grad.shape[2]))
    elif grad.ndim == 2:
        grad = np.reshape(grad, (grad.shape[0]*grad.shape[1]))

    updated_M = gradient_descent(bra, site, grad, activation_function)
    return updated_M


def compress(raw_state, bond_dim, threshold, activation_function):
    """ Right normalizes a compressed state then sweeps left->right
        and right->left until a minimum is reached
        i.e. the difference in our metrics between sweeps is less than a
        specified threshold

    Args:
        raw_state: MPS to be compressed
        bond_dim: Maximum bond dimension of compressed state
        threshold: Difference between sweeps under which a solution is found

    Returns:
        compressed_state: Final compressed state
        dist: List of overlap values after each sweep
        sim: List of scalar product values (cosine similarity) after each sweep
    """

    phys_dim = raw_state[0].shape[0]
    compressed_state = init.initialize_random_normed_state_MPS(len(raw_state),
                                                               bond_dim,
                                                               phys_dim)

    for tensor in compressed_state:
        if activation_function == 'linear':
            tensor = act.linear(tensor)
        elif activation_function == 'ReLU':  # XXX
            tensor = act.ReLU(tensor)
        elif activation_function == 'arctan':
            tensor = act.arctan(tensor)
        elif activation_function == 'tanh':
            tensor = act.tanh(tensor)
        elif activation_function == 'arcsinh':
            tensor = act.arcsinh(tensor)
        elif activation_function == 'sigmoid':
            tensor = act.sigmoid(tensor)
        elif activation_function == 'softplus':
            tensor = act.softplus(tensor)
        elif activation_function == 'SiLU':
            tensor = act.SiLU(tensor)
        elif activation_function == 'sinusoid':
            tensor = act.sinusoid(tensor)

    # Initialize accuracy metrics
    dist = []  # Frobenius norm
    sim = []   # Cosine similarity (Scalar product)
    dist.append(metrics.overlap(compressed_state, raw_state))
    sim.append(metrics.scalar_product(compressed_state, raw_state))
    # We sweep left to right and then back right to left across the mixed state
    while True:
        # Left->right sweep
        for site in range(0, len(raw_state)-1):
            compressed_state[site] = update_site(compressed_state, raw_state, activation_function,
                                                 site=site, dir='right')

        # Right->left sweep
        for site in range(len(raw_state)-1, 0, -1):
            compressed_state[site] = update_site(compressed_state, raw_state, activation_function,
                                                 site=site, dir='left')

        # Metrics are updated after each full sweep
        dist.append(metrics.overlap(compressed_state, raw_state))
        sim.append(metrics.scalar_product(compressed_state, raw_state))
        print(sim[-1])
        if np.abs(sim[-2]-sim[-1]) < threshold:
            break

    return compressed_state, dist, sim


    # if activation_function == 'ReLU':  # XXX
    #     updated_M = act.ReLU(updated_M)
    # elif activation_function == 'arctan':
    #     updated_M = act.arctan(updated_M)
    # elif activation_function == 'tanh':
    #     updated_M = act.tanh(updated_M)
    # elif activation_function == 'arcsinh':
    #     updated_M = act.arcsinh(updated_M)
    # elif activation_function == 'sigmoid':
    #     updated_M = act.sigmoid(updated_M)
    # elif activation_function == 'softplus':
    #     updated_M = act.softplus(updated_M)
    # elif activation_function == 'SiLU':
    #     updated_M = act.SiLU(updated_M)
    # elif activation_function == 'sinusoid':
    #     updated_M = act.sinusoid(updated_M)
