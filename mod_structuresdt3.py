import mpnum_v2 as mp
import numpy as np

"""
    In this module we construct useful structures derived from the mpnum library
"""

class MPSLIST():
    def __init__(self, N, Q, NB, BD):
        """
            List containing N*N mps
            
            :param N:   Number of electronical sites
            :param Q:   Number of local oscillators per site
            :param NB:  Dimensionality of local oscillator's Hilbert space
            :param BD:  Rank of the mps
        """
        self.list = np.array([mp.factory.zero(sites=N*Q, ldim=[(NB[q],) for q in range(Q)]*N, rank=BD)] * N**2)
        
class UNCMPSLIST():
    def __init__(self, N, Q, NB):
        """
            List containing N*N rank=1 mps
            
            :param N:   Number of electronical sites
            :param Q:   Number of local oscillators per site
            :param NB:  Dimensionality of local oscillator's Hilbert space
        """
        self.list = np.array([mp.factory.zero(sites=N*Q, ldim=[(NB[q]*NB[q],) for q in range(Q)]*N, rank=1)] * N**2)

 
class MPOLIST():
    def __init__(self, N, Q, NB):
        """
            List containing N*N rank=1 mpo
            
            :param N:   Number of electronical sites
            :param Q:   Number of local oscillators per site
            :param NB:  Dimensionality of local oscillator's Hilbert space
        """
        self.list = np.array([mp.factory.zero(sites=N*Q, ldim=[(NB[q]*NB[q],NB[q]*NB[q]) for q in range(Q)]*N, rank=1)] * N**2)
##########################################################################################


