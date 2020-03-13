# Title: DMRG for Bachelor Thesis
# Author: Aaron Sander
# Date: March 2020

# This program is used for initial learning of tensor network methods to be used
# in my bachelor thesis.
# It is an implementation of Matrix Product States (MPS) and Density Matrix
# Renormalization Group (DMRG) for finding the ground state of an arbitrary
# Hamiltonian

# Steps:
# 1. Initialize a Hamiltonian tensor for N particles
# 2. Decompose it with the MPS method
# 3. Apply variational method to determine energy value
# 4. Minimize energy one matrix at a time while holding others constant
# 5. Apply DMRG? (TBD)

import numpy as np

