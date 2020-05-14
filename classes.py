### CLASSES ###

import numpy as np
import math
from canonical_forms import *
from benchmarks import *
from contractions import *


class rawData:
    def __init__(self, bits, phys_dim):
        self.raw_data = np.random.rand(bits)
        self.norm = np.linalg.norm(self.raw_data)
        self.normed_data = self.raw_data / self.norm
        self.MPS = vector_to_left_canonical_MPS(self.normed_data, phys_dim, num_sites=int(math.log(bits, phys_dim)))


class compressedData:
    def __init__(self, uncompressed_MPS, phys_dim):
        # Creates a list of MPS's where each index i = maxBondDim-1
        self.MPS_list, self.sim = benchmark_dimensions(uncompressed_MPS,
                                                   phys_dim)


class randomState:
    def generate_MPS(self, num_sites, max_bond_dim, phys_dim):
        state = initialize_random_state(num_sites,
                                        max_bond_dim,
                                        phys_dim)
        state, _ = left_normalize(state)
        normed_state, _ = right_normalize(state)
        return normed_state

    def __init__(self, num_sites, max_bond_dim, phys_dim):
        self.MPS = self.generate_MPS(num_sites, max_bond_dim, phys_dim)


class Network:
    def __init__(self, MPS_1, MPO, MPS_2):
        self.bra = MPS_1
        self.MPO = MPO
        self.ket = MPS_2

        self.expectation_value = calculate_expectation(MPS_1, MPO, MPS_2,
                                                       vert_dir='down',
                                                       horiz_dir='right')
