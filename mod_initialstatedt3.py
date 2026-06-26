import numpy as np
import mpnum_v2 as mp
import sys

#-- Dampf modules ------------------------------------------------------------
from mod_parametersdt3 import N, Q, NB, KbT, Omega_osc, basis_check

"""
    In this module we compute the initial state, with a single excitation in 
    form of a purely electronical one sitting on the first site.
    Since there is no correlation, we deal with rank=1 mps.

"""
def state_initialization(focktoidtr):
    """
        Compute the initial state for the choosed configuration.
        
        :param focktoidtr:  Transformation from Fock to Idtr basis
    """
    locarr = mp.factory.zero(sites = Q, ldim = [(NB[q]*NB[q],) for q in range(Q)], rank = 1)
    for q in range(Q):
        if KbT[q] < 10e-6 and KbT[q] >= 0.:
            temp = np.zeros((1, NB[q]*NB[q], 1), dtype = complex)
            temp[0, 0, 0] = 1.
            locarr.lt.update(q, temp)
        elif KbT[q] >= 10e-6:
            Beta = 1./(KbT[q])
            temp = np.zeros((1, NB[q]*NB[q], 1), dtype = complex)
            for i in range(NB[q]):
                j = NB[q]*i+i
                term = np.exp(-Beta*Omega_osc[q]*i)
                temp[0,j,0] = term
            pfunc = 0.
            for i in range(100*NB[q]):
                pfunc += np.exp(-Beta*Omega_osc[q]*i)
            temp /= pfunc
            locarr.lt.update(q, temp)
        else:
            sys.exit('Temperature must be a positive float')
    #Chainlist identifies a list containing terms for all the M=NQ oscillators
    #Since all sites are equivalent, we need only to initialize Q tensors
    chainlist = []
    if basis_check == 'IdTr':
        locarridtr = mp.factory.zero(sites = Q, ldim = [(NB[q]*NB[q],) for q in range(Q)], rank = 1)
        for q in range(Q):
            temp = np.zeros((1, NB[q]*NB[q], 1), dtype = complex)
            for i in range(NB[q]*NB[q]):
                for j in range(NB[q]*NB[q]):
                    temp[0, i, 0] += locarr.lt[q][0, j, 0]*focktoidtr[q][j, i]
            locarridtr.lt.update(q, temp)
        for i in range(N):
            for q in range(Q):
                chainlist.append(locarridtr.lt[q])
        mps0 = mp.mparray.MPArray(chainlist)  
    elif basis_check == 'Fock':
        for i in range(N):
            for q in range(Q):
                chainlist.append(locarr.lt[q])
        mps0 = mp.mparray.MPArray(chainlist)
    else:
        sys.exit('Uncorrect value of basis_check')
    return mps0