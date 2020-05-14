import numpy as np


def scalar_product(A, B):
    """Calculates the scalar product of two Matrix Product States
       by contracting all positions vertically then horizontally.

    Args:
      A: first MPS
      B: second MPS

    Returns:
      result: Frobenius norm of A and B <A|B>
    """
    pos_0 = np.einsum('ij, ib->jb', A[0], B[0])
    pos_0 = np.reshape(pos_0, (A[0].shape[1]*B[0].shape[1]))
    result = pos_0
    for i in range(1, len(A)-1):
        pos_i = np.einsum('ijk, abk->iajb', A[i], B[i])
        pos_i = np.reshape(pos_i, (A[i].shape[0]*B[i].shape[0],
                                   A[i].shape[1]*B[i].shape[1]))
        result = np.einsum('i, ij->j', result, pos_i)
    pos_N = np.einsum('ij, ib->jb', A[-1], B[-1])
    pos_N = np.reshape(pos_N, (A[-1].shape[1]*B[-1].shape[1]))
    result = np.einsum('i, i', result, pos_N)
    return result


def overlap(A, B):
    """Calculates the Euclidean distance between two Matrix Product States
    Args:
      A: first MPS
      B: second MPS

    Returns:
      overlap: Euclidian distance ||State1 - State2||
    """
    overlap = np.sqrt(scalar_product(A, A) + scalar_product(B, B)
                      - scalar_product(A, B) - scalar_product(B, A))
    return overlap
