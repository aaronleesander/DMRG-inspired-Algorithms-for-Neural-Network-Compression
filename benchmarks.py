import numpy as np
import matplotlib.pyplot as plt
from initializations import *
from compression import *
from contractions import *


def test_canonical(MPS):
    """ Checks if each site in an MPS is left canonical or right canonical
        A tensor checked by contracting physical dimension and left bond
        B tensor checked by contracting physical dimension and right bond

        Output prints these checks, if the result is an identity matrix,
        then it is canonical.

        Final site returns the norm if all other sites are canonical.
    Args:
      MPS: list of tensors

    Returns:
      prints matrices after checking left and right canonicality
    """

    print("A Canonical Check \n")
    test_identity = np.einsum('ij, ib->jb', MPS[0], MPS[0])
    print("Pos", "0", ":\n", test_identity)
    for i in range(1, len(MPS)-1):
        test_identity = np.einsum('ijk, ibk->jb', MPS[i], MPS[i])
        print("Pos", i, ":\n", test_identity)
    test_identity = np.einsum('ij, ij', MPS[-1], MPS[-1])
    print("Pos", len(MPS)-1, ":\n", test_identity)

    print("\nB Canonical Check \n")
    test_identity = np.einsum('ij, ib->jb', MPS[-1], MPS[-1])
    print("Pos", len(MPS)-1, ":\n", test_identity)
    for i in range(len(MPS)-2, 0, -1):
        test_identity = np.einsum('ijk, ajk->ia', MPS[i], MPS[i])
        print("Pos", i, ":\n", test_identity)
    test_identity = np.einsum('ij, ij', MPS[0], MPS[0])
    print("Pos", 0, ":\n", test_identity)


def test_expectation_value_contractions(MPS, MPO):
    """ Tests if the result for the expectation value of an MPS and MPO
        is the same for all directions of contraction

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


def benchmark_dimensions(raw_state, phys_dim):
    """ Checks how well a raw state can be compressed for each possible
        max bond dimension up to the max bond dimension of the raw state

    Args:
      raw_state: MPS
      phys_dim: Physical dimension

    Returns:
        Plots Sweeps and Loss graphs
    """
    dimensional_states = []
    loss = []

    plt.figure()
    for bond_dim in range(1, int(phys_dim**np.floor(len(raw_state)/2))):
        compressed_state = initialize_random_state(len(raw_state), bond_dim, phys_dim)

        compressed_state, dist, sim = compress(compressed_state, raw_state, threshold=1e-8)
        dimensional_states.append(compressed_state)
        loss.append(100*(1-sim[-1]))

        # Plot sweeps and similarity for each bond dimension
        x = range(len(dist))
        plt.title("DMRG Compression (Bits = %d, L=%d)" % (phys_dim**len(raw_state), len(raw_state)))
        plt.xlabel("Sweeps")
        plt.ylabel("Cosine Similarity")
        plt.plot(x, sim, label="d=%d, CosSim=%f" % (bond_dim, 100*sim[-1]))  # TODO: Should use scatter plot
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
    return dimensional_states, sim


def benchmark_compression(raw_state, phys_dim, attempts):
    """ Checks average as well as upper and low bounds of loss at each
        compression dimension using a certain number of initial states

    Args:
      raw_state: MPS
      phys_dim: Physical dimension
      attempts: Total initial states at each compressed dimension to avoid
                local minima

    Returns:
        Plots Loss graphs
    """
    dimensional_states = []
    lower_bound_loss = []
    avg_loss = []
    upper_bound_loss = []

    # Try all bond dimensions under our raw state's max dimension
    for bond_dim in range(1, int(phys_dim**np.floor(len(raw_state)/2))):
        max_sim = 0
        min_sim = 10e6
        sim_values = []

        # Number of compressed states to try to avoid local minima
        for i in range(attempts):
            compressed_state = initialize_random_state(len(raw_state), bond_dim, phys_dim)

            compressed_state, dist, sim = compress(compressed_state, raw_state, threshold=1e-8)
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
