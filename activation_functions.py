import numpy as np
import math


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


    if activation_function == 'ReLU':  # XXX
        updated_M = act.ReLU(updated_M)
    elif activation_function == 'arctan':
        updated_M = act.arctan(updated_M)
    elif activation_function == 'tanh':
        updated_M = act.tanh(updated_M)
    elif activation_function == 'arcsinh':
        updated_M = act.arcsinh(updated_M)
    elif activation_function == 'sigmoid':
        updated_M = act.sigmoid(updated_M)
    elif activation_function == 'softplus':
        updated_M = act.softplus(updated_M)
    elif activation_function == 'SiLU':
        updated_M = act.SiLU(updated_M)
    elif activation_function == 'sinusoid':
        updated_M = act.sinusoid(updated_M)