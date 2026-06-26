import numpy           as np
import mpnum_v2           as mp

#-- Dampf modules ------------------------------------------------------------
from mod_parametersdt3 import N, Q, NB, epsilon, KbT, Omega_osc


def Uerelevant(Ue, std):
    """
        Compute the relevant indeces matrix of Ue and the associated relevant 
        coefficients for the non local evolution. 
        We consider j=0 a relevant index in order to avoid problems with 
        compressions of zero MPS (since initially only the j=0 term is non zero)
    
        :param Ue:       List of electronical evolution Ue operators
        :param std:      Single to double index mapping
    """
    rel_ind    = []
    rel_coeff = []
    for i in range(N*N):
        rel_ind.append([])
        rel_coeff.append([])
        
        # We inglobe an element with a non zero excitation (in here the first site)
        # to avoid having problem with relerr in svd (rewrite svd to use this effic.)
        term = Ue[std[i][0],std[0][0]]*np.conjugate(Ue[std[0][1],std[i][1]])
        rel_ind[i].append(0)
        rel_coeff[i].append(term)
        
        for j in range(1, N*N):
            term = Ue[std[i][0],std[j][0]]*np.conjugate(Ue[std[j][1],std[i][1]])
            if np.abs(term) >= epsilon:
                rel_ind[i].append(j)
                rel_coeff[i].append(term)
    return [rel_ind, rel_coeff]


#Works only in the IdTr basis
def adjoint_mapping():
    locten = [np.zeros((1, NB[q]**2, NB[q]**2, 1), dtype = complex) for q in range(Q)]
    for q in range(Q):
        #First basis' diagonal elements
        for i in range(NB[q]):
            locten[q][0, i, i, 0] = 1.
        # Now basis' off diagonal elements
        for i in range(NB[q], NB[q]**2):
            # 0 is even
            if ((i - NB[q]) % 2) == 0:
                locten[q][0, i, i, 0] = 1.
            else:
                locten[q][0, i, i, 0] = -1.
    return mp.mparray.MPArray(locten * N)


def Boltzmann_aver_num(q):
    """
        Computes the average occupation number, according to the
        Boltzmann distribution, of the oscillator q at the 
        temperature T
        
        :param T:           Temperature
        :param Omega_osc_q: Excitation energy of the local oscillator q
    """
    if KbT[q] <= 10e-6:
        return 0.
    else:
        return 1./(np.exp(Omega_osc[q]/(KbT[q]))-1.)