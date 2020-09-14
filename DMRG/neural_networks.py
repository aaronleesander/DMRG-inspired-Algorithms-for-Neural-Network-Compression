import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

import activation_functions as act
import canonical_forms as can
import compression as comp
import contractions as con
import initializations as init
import metrics


def calculate_params(data):
    """ Calculates total number of parameters in a layer

    Args:
        data: List of tensors or any structure (MPS, MPO, vector, etc.)

    Returns:
        params: Total number of elements in data
    """
    params = 0
    for tensor in data:
        params += tensor.size

    return params


def weights_to_MPO(weights, D, sigma, sigma_prime, layer):
    """ Used specifically to convert the weights output by the MPO-Net networks into their MPO form

    Args:
        weights: List of weights matrices including biases
        sigma: List of input physical dimensions
        sigma_prime: List of output physical dimensions
        layer: Layer number (indexed from 0) such that we start at the correct weights matrix

    Returns:
        MPO: List of tensors corresponding to MPO of layer
        bias: Vector corresponding to the bias found in the list of weights
    """
    # Reshape weights, order of indices are an assumption and may need to be modified
    MPO = []
    num_sites = len(sigma)
    starting_site = layer*(num_sites+1)  # +1 to skip bias

    for i in range(num_sites):
        # sigma'*D_right, D_left*sigma
        if i == 0:
            site = np.reshape(weights[i+starting_site], (sigma_prime[i], D, sigma[i]))
            # Need shape (D_right, sigma, sigma')
            site = np.transpose(site, (1, 2, 0))
        elif i != 0 and i != num_sites-1:
            # First D is right bond, Second D is left bond
            site = np.reshape(weights[i+starting_site], (sigma_prime[i], D, D, sigma[i]))
            # Need shape (D_left, D_right, sigma, sigma')
            site = np.transpose(site, (2, 1, 3, 0))
        elif i == num_sites-1:
            # First D is right bond, Second D is left bond
            site = np.reshape(weights[i+starting_site], (sigma_prime[i], D, sigma[i]))
            # Need shape (D_left, sigma, sigma')
            site = np.transpose(site, (1, 2, 0))
        MPO.append(site)
    bias = weights[num_sites+starting_site]
    return MPO, bias


def open_legs(MPS, sigma, sigma_prime, bond_dim):
    """ Converts an MPS to an MPO by opening the physical dimensions

    Args:
        MPS: List of tensors of MPS
        sigma: List of input physical dimensions
        sigma_prime: List of output physical dimensions
        bond_dim: List of bond dimensions

    Returns:
        MPO: List of tensors of MPO with given dimensions
    """
    MPO = []
    for i, site in enumerate(MPS):
        if i == 0 or i == len(MPS)-1:
            site = np.reshape(site.T, (bond_dim[i], sigma[i], sigma_prime[i]))
        else:
            site = np.reshape(site, (bond_dim[i-1], bond_dim[i], sigma[i], sigma_prime[i]))
        MPO.append(site)
    return MPO


def close_legs(MPO):
    """ Converts an MPO to an MPS by closing physical dimensions

    Args:
        MPO: List of tensors of MPO

    Returns:
        MPS: List of tensors of MPS with combined physical dimensions of MPO
    """
    MPS = []
    for i, site in enumerate(MPO):
        if i == 0 or i == len(MPO)-1:
            site = np.reshape(site, (site.shape[0], site.shape[1]*site.shape[2])).T
        else:
            site = np.reshape(site, (site.shape[0], site.shape[1], site.shape[2]*site.shape[3]))
        MPS.append(site)
    return MPS


def compress_layer(raw_state, phys_dim, threshold, compressed_state=0, plot=0):
    """ Initializes a compressed state then sweeps left->right
        and right->left until a minimum is reached
        i.e. the difference in our metrics between sweeps is less than a
        specified threshold up to the bond dimension of the raw state

        Note: Does NOT normalize the state unlike the compression function
              found in compression.py

    Args:
        raw_state: MPS to be compressed
        phys_dim: List of physical dimensions by site
        threshold: Difference between sweeps under which a solution is found
        compressed_state: Initial starting state if necessary
                          otherwise a random MPS is ini
        plot: Whether or not to plot the compression values (0 off, 1 on)

    Returns:
        compressions: Final compressed state at each bond dimension
        best_dist: List of overlap values for each bond dimension
        best_sim: List of cosine similarity values for each bond dimension
    """
    if compressed_state == 0:
        compressed_state = init.initialize_random_MPS_with_changing_phys_dim(phys_dim,
                                                                             num_sites=len(raw_state),
                                                                             bond_dim=1)
    bond_dim_raw_state = raw_state[math.ceil(len(raw_state)/2)].shape[0]
    max_bond_dim = 1

    # Initialize accuracy metrics
    dist = []  # Frobenius norm
    sim = []   # Cosine similarity (Scalar product)
    dist.append(metrics.overlap(compressed_state, raw_state))
    sim.append(metrics.scalar_product(compressed_state, raw_state))
    best_dist = []
    best_sim = []
    compressions = []
    # We sweep left to right and then back right to left across the mixed state
    while True:
        # Left->right sweep
        for site in range(0, len(raw_state)-1):
            compressed_state[site], compressed_state[site+1] = comp.update_site(compressed_state, raw_state,
                                                                           site=site, dir='right')
        # Right->left sweep
        for site in range(len(raw_state)-1, 0, -1):
            compressed_state[site], compressed_state[site-1] = comp.update_site(compressed_state, raw_state,
                                                                           site=site, dir='left')

        # Metrics taken after each sweep
        dist.append(metrics.overlap(compressed_state, raw_state))
        sim.append(metrics.similarity(compressed_state, raw_state))
        # Check if sweeps are still working
        if np.abs(dist[-2]-dist[-1]) < threshold:
            # Normalize to maintain length and update metrics
            #compressed_state, _ = can.left_normalize(compressed_state)
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

        plt.figure()
        plt.title("Cosine Similarity vs. Euclidean Distance")
        plt.xlabel("Euclidean Distance")
        plt.ylabel("Cosine Similarity")

        plt.plot(best_dist, best_sim)

    return compressions, best_dist, best_sim


def compress_layer_single_dim(raw_state, phys_dim, threshold):
    """ Initializes a compressed state then sweeps left->right
        and right->left until a minimum is reached at a single bond dimension

        Note: Does NOT normalize the state unlike the compression function
              found in compression.py

    Args:
        raw_state: MPS to be compressed
        phys_dim: List of physical dimensions by site
        threshold: Difference between sweeps under which a solution is found

    Returns:
        compressed_state: List of MPS tensors for final compressed state
        dist: List of overlap values for each bond dimension
        sim: List of cosine similarity values for each bond dimension
    """
    compressed_state = init.initialize_random_MPS_with_changing_phys_dim(phys_dim,
                                                                         num_sites=len(raw_state),
                                                                         bond_dim=compressed_dim)
    bond_dim_raw_state = raw_state[math.ceil(len(raw_state)/2)].shape[0]

    # Initialize accuracy metrics
    dist = []  # Frobenius norm
    sim = []   # Cosine similarity (Scalar product)
    dist.append(metrics.overlap(compressed_state, raw_state))
    sim.append(metrics.similarity(compressed_state, raw_state))
    # We sweep left to right and then back right to left across the mixed state
    while True:
        # Left->right sweep
        for site in range(0, len(raw_state)-1):
            compressed_state[site], compressed_state[site+1] = comp.update_site(compressed_state, raw_state,
                                                                           site=site, dir='right')
        # Right->left sweep
        for site in range(len(raw_state)-1, 0, -1):
            compressed_state[site], compressed_state[site-1] = comp.update_site(compressed_state, raw_state,
                                                                           site=site, dir='left')

        # Metrics taken after each sweep
        dist.append(metrics.overlap(compressed_state, raw_state))
        sim.append(metrics.similarity(compressed_state, raw_state))
        # Check if sweeps are still working
        if np.abs(dist[-2]-dist[-1]) < threshold:
            print("Sim:", sim[-1], "Dist:", dist[-1], "BondDim:", compressed_dim)
            break

    return compressed_state, dist, sim


def FC2(MPO_0, bias_0, MPO_1, bias_1):
    """ Recreation of the FC2 network

    Args:
        MPO_0: List of tensors corresponding to MPO of 0th layer
        bias_0: Vector corresponding to bias of 0th layer
        MPO_1: List of tensors corresponding to MPO of 0th layer
        bias_1: Layer number (indexed from 0) such that we start at the correct weights matrix

    Returns:
        acc: Accuracy on test set
        t: Time to contract all layers
    """
    data = input_data.read_data_sets("./data/", validation_size=0, one_hot=True)

    start = time.time()

    for i in range(len(MPO_0)-1):
        if i == 0:
            layer_0 = con.contract_horizontal(MPO_0[i], MPO_0[i+1], 'right')
        else:
            layer_0 = con.contract_horizontal(layer_0, MPO_0[i+1], 'right')

    for i in range(len(MPO_0)-1):
        if i == 0:
            layer_1 = con.contract_horizontal(MPO_1[i], MPO_1[i+1], 'right')
        else:
            layer_1 = con.contract_horizontal(layer_1, MPO_1[i+1], 'right')
    layer_0 = np.reshape(layer_0, (784, 256))
    layer_1 = np.reshape(layer_1, (256, 10))

    end = time.time()

    total = 10000
    correct = 0
    for i in range(0, total):
        t0 = time.time()
        xW_1 = data.test.images[i] @ layer_0 + bias_0
        xW_1 = act.ReLU(xW_1)
        result = xW_1 @ layer_1 + bias_1
        index = result.argmax()
        if index == list(data.test.labels[i]).index(1):
            correct += 1

    acc = correct/total*100
    t = end-start
    return acc, t


def vector_to_left_canonical_MPS_NN(tensor, phys_dim):
    """ Decomposes a vector into a left-canonical MPS with changing
        physcial dimensions. Final site will not be canonical due to
        original norm.

    Args:
        tensor: Original flattened tensor (vector)
        phys_dim: List of physical dimension necessary on MPS at each site
        num_sites: number of sites necessary (L)

    Returns:
        A_tensors: Left canonical MPS of input tensor
    """

    A_tensors = []
    num_sites = len(phys_dim)
    for i in range(0, num_sites-1):
        # Remove one leg such that tensor has shape (d, d^(L-1)) with L sites
        if i == 0:
            reshaped_tensor = np.reshape(tensor, (phys_dim[i],
                                                  tensor.shape[0]//phys_dim[i]))
        else:
            reshaped_tensor = np.reshape(tensor, (A_tensors[-1].shape[1]*phys_dim[i],
                                                  tensor.shape[1]//phys_dim[i]))       
        #print("Reshape:", reshaped_tensor.shape)

        # SVD and save the rank for the next iteration of the loop
        U, S_vector, V = np.linalg.svd(reshaped_tensor, full_matrices=False)
        rank = len(S_vector)

        if i == 0:
            # No need to reshape since U is already a left-canonical matrix
            A_tensors.append(U)
        else:
            # Break apart first leg of U into a left bond dimension
            # and physical dimension
            U = np.reshape(U, (A_tensors[-1].shape[1], phys_dim[i], U.shape[1]))
            # Transpose so that we have the correct shape
            # (left bond, right bond, physical dimension)
            U = np.transpose(U, (0, 2, 1))
            A_tensors.append(U)

        # We recreate the tensor with the remaining legs
        tensor = np.diag(S_vector) @ V
    # Final A tensor is the remaining tensor after all other legs removed
    A_tensors.append(tensor.T)

    return A_tensors
