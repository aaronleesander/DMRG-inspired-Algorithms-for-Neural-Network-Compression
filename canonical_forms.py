### CANONICAL FORMS ###

import numpy as np


### Normalizes the tensor network such that A_dagger A = Identity ###
def left_normalize(MPS):
    lambda_tensors = []
    # Loop from left bound to right bound
    for i in range(0, len(MPS)):
        # Left bound (2 x d)
        if i == 0:
            M = MPS[i]
            # Dimensions (d x d), (d x 2), (2 x 2)
            U, S_vector, V = np.linalg.svd(M, full_matrices=False)
            S = np.diag(S_vector)

            MPS[i] = U
            # Contract such that M' = SVM
            MPS[i+1] = np.einsum('ij, jbc->ibc', S @ V, MPS[i+1])
        # Inner tensors (d x d x 2)
        elif i != len(MPS)-1:
            # Reshape such that (d x d x 2) -> (d x 2 x d) for correct reshape
            # Legs closest to each other collape
            M = np.transpose(MPS[i], (0, 2, 1))
            # Collapse left bond and physical dimensions
            M = np.reshape(M, (MPS[i].shape[0]*MPS[i].shape[2], MPS[i].shape[1]))
            U, S_vector, V = np.linalg.svd(M, full_matrices=False)
            S = np.diag(S_vector)

            # Dimensions according to d x 2 x d still
            # (Left bond, physical dimension, right bond)
            MPS[i] = np.reshape(U, (MPS[i].shape[0], MPS[i].shape[2], U.shape[1]))
            MPS[i] = np.transpose(MPS[i], (0, 2, 1))  # Reshape back to (d x d x 2)

            # Last tensor is a rank-2 tensor
            if i == len(MPS)-2:
                # Transpose due to convention of using 2 x d for last position
                MPS[i+1] = np.einsum('ij, jb->ib', S @ V, MPS[i+1].T).T
            # Other inner tensors are rank-3 tensors
            else:
                MPS[i+1] = np.einsum('ij, jbc->ibc', S @ V, MPS[i+1])

        # Right bound (2 x d)
        elif i == len(MPS)-1:
            # No right bond dimension exists, so we set it to 1
            M = np.reshape(MPS[i], (MPS[i].shape[0]*MPS[i].shape[1], 1))
            U, S_vector, V = np.linalg.svd(M, full_matrices=False)
            S = np.diag(S_vector)
            MPS[i] = np.reshape(U, (MPS[i].shape[0], MPS[i-1].shape[1]))
        lambda_tensors.append(S)

    lambda_tensors.pop(-1)  # Last element is just the norm
    return MPS, lambda_tensors


### Normalizes the tensor network such that B B_dagger = Identity ###
def right_normalize(MPS):
    lambda_tensors = []
    # Loop from right bound to left bound
    for i in range(len(MPS)-1, -1, -1):
        # Right bound (2 x d)
        if i == len(MPS)-1:
            M = MPS[i].T  # Needs to be (d x 2)
            # Dimensions (d x d) (d x 2) (2 x 2)
            U, S_vector, V = np.linalg.svd(M, full_matrices=False)
            S = np.diag(S_vector)
            MPS[i] = V.T  # Transpose so that bond dimension is second
            # Update next position M' = MUS
            MPS[i-1] = np.einsum('ijk, jb->ibk', MPS[i-1], U @ S)

        # Inner tensor
        elif i != 0:
            # Collapse right bond and physical dimension (no need to permute)
            M = np.reshape(MPS[i], (MPS[i].shape[0], MPS[i].shape[1]*MPS[i].shape[2]))

            U, S_vector, V = np.linalg.svd(M, full_matrices=False)
            S = np.diag(S_vector)

            # Reshape to (left bond, right bond/previous left bond, physical dim)
            if i == len(MPS)-2:  # Last site does not have left bond in first spot
                MPS[i] = np.reshape(V, (V.shape[0], MPS[i+1].shape[1], MPS[i].shape[2]))
            else:
                MPS[i] = np.reshape(V, (V.shape[0], MPS[i+1].shape[0], MPS[i].shape[2]))

            # Last tensor is a rank-2 tensor
            if i == 1:
                MPS[i-1] = np.einsum('ij, jk->ik', MPS[i-1], U @ S)
            # Other inner tensors are rank-3 tensors
            else:
                MPS[i-1] = np.einsum('ijk, jb->ibk', MPS[i-1], U @ S)

        # Left bound
        elif i == 0:
            M = np.reshape(MPS[i], (1, MPS[i].shape[0]*MPS[i].shape[1]))
            U, S_vector, V = np.linalg.svd(M, full_matrices=False)
            S = np.diag(S_vector)
            MPS[i] = np.reshape(V, (MPS[i].shape[0], MPS[i+1].shape[0]))
        lambda_tensors.append(S)

    lambda_tensors.pop(-1)  # Last element is just the norm
    # Lambdas were appened from right, so list needs to be reversed
    return MPS, lambda_tensors[::-1]


# Decomposes a vector of length d^L (phys_dim**num_sites) into a #
# left-canonical MPS. Final site will not be canonical due to original norm #
def vector_to_left_canonical_MPS(tensor, phys_dim, num_sites):
    # Initialize MPS of A tensors, rank set to 1 for first calculation to work in loop
    A_tensors = []
    rank = 1

    # We loop over each leg and decompose it into its own tensor
    for i in range(1, num_sites):
        # Remove one leg such that tensor has shape (d, d^(L-1)) with L sites
        reshaped_tensor = np.reshape(tensor, (rank*phys_dim, phys_dim**(num_sites-i)))
        # Decompose it and save the rank for the next iteration of the loop
        U, S_vector, V = np.linalg.svd(reshaped_tensor, full_matrices=False)
        rank = len(S_vector)

        if i == 1:
            # No need to reshape since U is already a left-canonical matrix (2 x d)
            A_tensors.append(U)
        else:
            # Break apart first leg of U into a left bond dimension and physical dimension
            U = np.reshape(U, (A_tensors[-1].shape[1], phys_dim, U.shape[1]))
            # Transpose so that we have the correct shape (left bond, right bond, physical dimension)
            U = np.transpose(U, (0, 2, 1))
            A_tensors.append(U)

        # We recreate the remaining tensor and then reshape it to further decompose
        tensor = np.diag(S_vector) @ V

    # Final tensor is the remaining tensor after all other legs removed
    A_tensors.append(tensor)
    return A_tensors


# ### Creates a bond canonical form with left and right normalization on each side of bond ###
def bond_canonical(gamma_tensors, lambda_tensors, bond):
    # Bond i occurs between site i, i+1
    A_tensors = []
    B_tensors = []

    for i in range(0, len(gamma_tensors)):
        if i < bond:
            if i == 0:
                A = gamma_tensors[0]
            else:
                A = np.einsum('ij, jbc->ibc', lambda_tensors[i-1], gamma_tensors[i])
            A_tensors.append(A)
        elif i >= bond:
            if i == len(gamma_tensors)-1:
                B = gamma_tensors[-1]
            else:
                B = np.einsum('ijk, jb->ibk', gamma_tensors[i], lambda_tensors[i])
            B_tensors.append(B)
    MPS = A_tensors + B_tensors
    return MPS


### Creates a site canonical form such as AAAA...M...BBBB ###
def site_canonical(gamma_tensors, lambda_tensors, site):
    # Bond i occurs between site i and i+1
    A_tensors = []
    B_tensors = []

    # Create left canonical matrices where A = Lambda*Gamma
    # First site is simply A = Gamma (Lambda = 1 for normed state)
    for i in range(0, len(gamma_tensors)):
        if i < site:
            if i == 0:
                A = gamma_tensors[0]
            else:
                A = np.einsum('ij, jbc->ibc', lambda_tensors[i-1], gamma_tensors[i])
            A_tensors.append(A)

    # Create matrix for optimization where M = Lambda*Gamma*Lambda
    # End sites have only one lambda
        elif i == site:
            if i == 0:
                M = np.einsum('ij, jb->ib', gamma_tensors[i], lambda_tensors[i])
            elif i == len(gamma_tensors)-1:
                M = np.einsum('ij, aj->ai', lambda_tensors[i-1], gamma_tensors[i])
            else:
                M = np.einsum('ij, jbc, bn->inc', lambda_tensors[i-1], gamma_tensors[i], lambda_tensors[i])

    # Create right canonical matrices where B = Gamma*Lambda
    # Last site is simply B = Gamma (Lambda = 1 for normed states)
        elif i > site:
            if i == len(gamma_tensors)-1:
                B = gamma_tensors[-1]
            else:
                B = np.einsum('ijk, jb->ibk', gamma_tensors[i], lambda_tensors[i])
            B_tensors.append(B)

    MPS = A_tensors + [M] + B_tensors
    return MPS

### Decomposes our MPS into Gamma matrices on sites, Lambda matrices on bonds ###
### Lambda matrices contain the singular values, final lambda matrix is related to norm ###
### Requires a fully left or right normalized state ###
### FINAL GAMMA IS ALSO 2 x d ###


def vidal_notation(tensors, lambda_tensors, normalization):
    # Trim singular values under a threshold, otherwise inverse is hard to calculate
    threshold = 10e-8
    for i in range(0, len(lambda_tensors)):
        lambda_tensors[i][lambda_tensors[i] < threshold] = 0

    # Calculate inverse matrices
    lambda_inverse = []
    for i in range(0, len(lambda_tensors)):
        try:
            lambda_inverse.append(np.array(np.linalg.inv(lambda_tensors[i])))
        except LinAlgError:
            print("Lambda tensor has no inverse.")
            print(lambda_tensors[i])

    if normalization == 'left':  # A tensors as input
        A_tensors = tensors[:]
        # Calculate gamma tensors
        gamma_tensors = []
        gamma_tensors.append(A_tensors[0])  # First gamma is just the first A
        for i in range(1, len(A_tensors)-1):
            gamma = np.einsum('ij, jbc->ibc', lambda_inverse[i-1], A_tensors[i])
            gamma_tensors.append(gamma)
        gamma = np.einsum('ij, aj->ai', lambda_inverse[-1], A_tensors[-1])
        gamma_tensors.append(gamma)

    elif normalization == 'right':  # B tensors as input
        B_tensors = tensors[:]
        # Calculate gamma tensors
        gamma_tensors = []
        gamma = np.einsum('ij, jb->ib', B_tensors[0], lambda_inverse[0])
        gamma_tensors.append(gamma)
        for i in range(1, len(B_tensors)-1):
            gamma = np.einsum('ijk, jb->ibk', B_tensors[i], lambda_inverse[i])
            gamma_tensors.append(gamma)
        gamma_tensors.append(B_tensors[-1])  # Last gamma is just the last B

    return gamma_tensors, lambda_tensors
