import matplotlib.pyplot as plt
import math
import numpy as np

import activation_functions as act
import canonical_forms as can
import initializations as init
import metrics
import neural_networks as nn

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


def update_site(bra, ket, site, dir):
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

    updated_M = contract_environment(bra, ket, site)

    # For a left->right sweep, similar to left normalization
    if dir == 'right':
        # Inner tensor needs to be reshaped
        if updated_M.ndim == 3:
            reshaped_M = np.transpose(updated_M, (0, 2, 1))  # Move left bond and physical dimension together
            reshaped_M = np.reshape(reshaped_M, (reshaped_M.shape[0]*reshaped_M.shape[1],
                                                 reshaped_M.shape[2]))
            U, S_vector, V = np.linalg.svd(reshaped_M, full_matrices=False)
            A_tensor = np.reshape(U, (bra[site].shape[0],
                                      bra[site].shape[2],
                                       U.shape[1]))
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


def compress(raw_state, threshold, compressed_state=0, plot=0):
    """ Right normalizes a compressed state then sweeps left->right
        and right->left until a minimum is reached
        i.e. the difference in our metrics between sweeps is less than a
        specified threshold up to the bond dimension of the raw state

    Args:
        raw_state: MPS to be compressed
        threshold: Difference between sweeps under which a solution is found
        compressed_state: Initial starting state if necessary
        plot: Whether or not to plot the compression values (0 off, 1 on)

    Returns:
        compressions: Final compressed state at each bond dimension
        best_dist: List of overlap values for each bond dimension
        best_sim: List of cosine similarity values for each bond dimension
    """

    bond_dim_raw_state = raw_state[math.ceil(len(raw_state)/2)].shape[0]
    max_bond_dim = 1
    if compressed_state == 0:
        compressed_state = init.initialize_random_normed_state_MPS(len(raw_state),
                                                                bond_dim=max_bond_dim,
                                                                phys_dim=raw_state[0].shape[0])

    # Initialize accuracy metrics
    dist = []  # Frobenius norm
    sim = []   # Cosine similarity (Scalar product)
    dist.append(metrics.overlap(compressed_state, raw_state))
    sim.append(metrics.similarity(compressed_state, raw_state))
    best_dist = []
    best_sim = []
    compressions = []
    # We sweep left to right and then back right to left across the mixed state
    while True:
        # Left->right sweep
        for site in range(0, len(raw_state)-1):
            compressed_state[site], compressed_state[site+1] = update_site(compressed_state, raw_state,
                                                                           site=site, dir='right')

        # Right->left sweep
        for site in range(len(raw_state)-1, 0, -1):
            compressed_state[site], compressed_state[site-1] = update_site(compressed_state, raw_state,
                                                                           site=site, dir='left')

        # Metrics taken after each sweep
        dist.append(metrics.overlap(compressed_state, raw_state))
        sim.append(metrics.similarity(compressed_state, raw_state))

        # Check if sweeps are still working
        if np.abs(dist[-2]-dist[-1]) < threshold:
            # Normalize to maintain length and update metrics
            compressed_state, _ = can.left_normalize(compressed_state)
            best_dist.append(dist[-1])
            best_sim.append(sim[-1])
            print("Sim:", best_sim[-1], "Dist:", best_dist[-1], "BondDim:", max_bond_dim)
            compressions.append(compressed_state[:])

            # Break if we cannot increase bond dimension anymore
            if max_bond_dim+1 == bond_dim_raw_state:
                break

            # Break if changing bond dimension did not do enough
            if len(best_dist) > 1 and np.abs(best_dist[-2]-best_dist[-1] < threshold):
                break

            # Update each tensor by increasing bond dimension
            for i, tensor in enumerate(compressed_state):
                if tensor.ndim == 2:
                    new_tensor = np.zeros((tensor.shape[0], tensor.shape[1]+1))
                    new_tensor[:tensor.shape[0], :tensor.shape[1]] = tensor
                    compressed_state[i] = new_tensor

                elif tensor.ndim == 3:
                    new_tensor = np.zeros((tensor.shape[0]+1, tensor.shape[1]+1, tensor.shape[2]))
                    new_tensor[:tensor.shape[0], :tensor.shape[1], :tensor.shape[2]] = tensor
                    compressed_state[i] = new_tensor
            max_bond_dim = compressed_state[math.ceil(len(compressed_state)/2)].shape[0]

    if plot == 1:
        max_bond_dim = range(1, len(best_dist)+1)
        fig, ax1 = plt.subplots()

        color = 'tab:blue'
        ax1.set_xlabel('Compressed Dimension')
        ax1.set_ylabel('Cosine Similarity', color=color)
        ax1.plot(max_bond_dim, best_sim, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Euclidean Distance', color=color)
        ax2.plot(max_bond_dim, best_dist, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Metrics vs. Compressed Dimension')

        fig.tight_layout()
        plt.show()

        # plt.figure()
        # plt.title("Cosine Similarity vs. Max Bond Dimension")
        # plt.xlabel("Max Bond Dimension")
        # plt.ylabel("Cosine Similarity")

        # max_bond_dim = range(1, len(best_dist)+1)
        # plt.plot(max_bond_dim, best_sim)

        # plt.figure()
        # plt.title("Euclidean Distance vs. Max Bond Dimension")
        # plt.xlabel("Max Bond Dimension")
        # plt.ylabel("Euclidean Distance")
        # plt.plot(max_bond_dim, best_dist)

    return compressions, best_dist, best_sim


def benchmark_compression(raw_state, threshold):
    """ Checks how well a raw state can be compressed for each possible
        max bond dimension up to the max bond dimension of the raw state

    Args:
        raw_state: MPS to be compressed
        threshold: Difference between sweeps under which a solution is found

    Returns:
        compressions: Compressed state for each bond dimension
                      such that the indexing compressions[i] has a
                      max bond dimension i+1
        sim: Cosine similarity for each corresponding compressed state
        loss: Percentage loss for each corresponding compressed state
        Plots Sweeps and Loss graphs
    """

    compressions = []
    loss = []
    phys_dim = raw_state[0].shape[0]
    plt.figure()
    for bond_dim in range(1, int(phys_dim**np.floor(len(raw_state)/2))):

        compressed_state, dist, sim = compress(raw_state, bond_dim, threshold)
        compressions.append(compressed_state)
        loss.append(100*(1-sim[-1]))

        # Plot sweeps and similarity for each bond dimension
        x = range(len(dist))
        plt.title("DMRG Compression (Bits = %d, L=%d)" % (phys_dim**len(raw_state), len(raw_state)))
        plt.xlabel("Sweeps")
        plt.ylabel("Euclidean Distance")
        plt.plot(x, dist, label="d=%d, Dist=%f" % (bond_dim, dist[-1]))  # TODO: Should use scatter plot
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Plot loss vs. bond dimension and its bounds
    plt.figure()
    plt.title("Percentage Loss vs. Max Bond Dimension (Bits=%d)" % phys_dim**len(raw_state))
    plt.xlabel("Max Bond Dimension")
    plt.ylabel("Loss (%)")
    plt.plot(range(1, len(loss)+1), loss, label='Loss')
    plt.legend()

    # Marker at index where we have less than 5% loss
    try:
        index = next(x for x, value in enumerate(loss) if value < 5)+1
        plt.axvline(index, color='r', linestyle='--')
        plt.text(index+0.1, max(loss)/2, '5% Loss Threshold', color='r')
        plt.text(index+0.1, max(loss)/2-0.1*max(loss), 'Dim = %d' % index, color='r')
    except StopIteration:
        print("No loss better than 5%")
    return compressions, sim, loss


def benchmark_compression_loss(raw_state, attempts):
    """ Checks average as well as upper and low bounds of loss at each
        compression dimension using a certain number of initial states

    Args:
        raw_state: MPS
        attempts: Total initial states at each compressed dimension to avoid
                local minima

    Returns:
        Plots Loss graphs
    """
    dimensional_states = []
    lower_bound_loss = []
    avg_loss = []
    upper_bound_loss = []
    phys_dim = raw_state[0].shape[0]
    # Try all bond dimensions under our raw state's max dimension
    for bond_dim in range(1, int(phys_dim**np.floor(len(raw_state)/2))):
        max_sim = 0
        min_sim = 10e6
        sim_values = []

        # Number of compressed states to try to avoid local minima
        for i in range(attempts):
            compressed_state = init.initialize_random_normed_state_MPS(len(raw_state), bond_dim, phys_dim)

            compressed_state, dist, sim = compress(raw_state, bond_dim, threshold=1e-8)
            sim_values.append(sim[-1])

            # We want to save the min, avg, and max loss for a given bond dimension
            if sim[-1] > max_sim:
                max_sim = sim[-1]
            if sim[-1] < min_sim:
                min_sim = sim[-1]
        avg_loss.append(100*(1-np.mean(sim_values)))
        lower_bound_loss.append(100*(1-max_sim))
        upper_bound_loss.append(100*(1-min_sim))

    # Plot loss vs. bond dimension and its bounds
    plt.figure()
    plt.title("Percentage Loss vs. Max Bond Dimension (Bits=%d)" % phys_dim**len(raw_state))
    plt.xlabel("Max Bond Dimension")
    plt.ylabel("Loss (%)")
    plt.plot(range(1, len(avg_loss)+1), avg_loss, label='Average Loss')
    plt.plot(range(1, len(avg_loss)+1), lower_bound_loss, label='Minimum Loss')
    plt.plot(range(1, len(avg_loss)+1), upper_bound_loss, label='Maximum Loss')
    plt.legend()

    # Marker at index where we have less than 5% loss
    try:
        index = next(x for x, value in enumerate(avg_loss) if value < 5)+1
        plt.axvline(index, color='r', linestyle='--')
        plt.text(index+0.1, max(avg_loss)/2, '5% Loss Threshold', color='r')
        plt.text(index+0.1, max(avg_loss)/2-0.1*max(avg_loss), 'Dim = %d' % index, color='r')
    except StopIteration:
        print("No loss better than 5%")
