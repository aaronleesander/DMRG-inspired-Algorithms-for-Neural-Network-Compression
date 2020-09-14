import math
import numpy as np


def ReLU(tensor):
    """ Applies a ReLU function to a tensor

    Args:
        tensor

    Returns:
        tensor
    """

    tensor[tensor < 0] = 0

    return tensor


def arctan(tensor):
    """ Applies an arctan
     function to a tensor

    Args:
        tensor

    Returns:
        tensor
    """

    tensor = np.arctan(tensor)

    return tensor


def tanh(tensor):
    """ Applies an arctan
     function to a tensor

    Args:
        tensor

    Returns:
        tensor
    """

    tensor = np.tanh(tensor)

    return tensor


def arcsinh(tensor):
    """ Applies an arctan
     function to a tensor

    Args:
        tensor

    Returns:
        tensor
    """

    tensor = np.arcsinh(tensor)

    return tensor


def sigmoid(tensor):
    """ Applies an arctan
     function to a tensor

    Args:
        tensor

    Returns:
        tensor
    """

    return 1 / (1 + np.exp(-tensor))


def softplus(tensor):
    """ Applies an arctan
     function to a tensor

    Args:
        tensor

    Returns:
        tensor
    """

    return np.log(1 + np.exp(tensor))


def SiLU(tensor):
    """ Applies an arctan
     function to a tensor

    Args:
        tensor

    Returns:
        tensor
    """

    return tensor / (1 + np.exp(tensor))


def sinusoid(tensor):
    """ Applies an arctan
     function to a tensor

    Args:
        tensor

    Returns:
        tensor
    """

    return np.sin(tensor)


def linear(tensor):
    """ Applies an arctan
     function to a tensor

    Args:
        tensor

    Returns:
        tensor
    """

    return tensor
