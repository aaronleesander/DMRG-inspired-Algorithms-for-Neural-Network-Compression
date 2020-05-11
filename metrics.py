### METRICS ###

import numpy as np

### TODO: Should be supplemented by a matrix norm of reshaped vectors
### Calculates scalar product/overlap between two states ###
### <A|B> = N_a*N_b or <A|A> = N^2 = 1/S^2 ###


def scalar_product(A, B):
    # Each position is collapsed vertically
    # Then the results are built by collapsing horizontally
    pos_0 = np.einsum('ij, ib->jb', A[0], B[0])
    pos_0 = np.reshape(pos_0, (A[0].shape[1]*B[0].shape[1]))
    result = pos_0
    for i in range(1, len(A)-1):
        pos_i = np.einsum('ijk, abk->iajb', A[i], B[i])
        pos_i = np.reshape(pos_i, (A[i].shape[0]*B[i].shape[0], A[i].shape[1]*B[i].shape[1]))
        result = np.einsum('i, ij->j', result, pos_i)
    pos_N = np.einsum('ij, ib->jb', A[-1], B[-1])
    pos_N = np.reshape(pos_N, (A[-1].shape[1]*B[-1].shape[1]))
    result = np.einsum('i, i', result, pos_N)
    return result


### Calculates the distance between two states ###
def overlap(A, B):
    overlap = np.sqrt(scalar_product(A, A) + scalar_product(B, B)
                      - scalar_product(A, B) - scalar_product(B, A))
    return overlap
