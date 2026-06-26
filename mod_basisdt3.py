import numpy        as np
import scipy.linalg as spla
import sys

#-- Dampf modules ------------------------------------------------------------
from mod_parametersdt3 import N, NB, Q

"""
    Brief description: 
        
    In this module we initialize the basis matrices, the mapping between different 
    basis and the mapping between single and double index notation
"""

def basis_initialization(bas = '', vectorized = False):
    """
        Initialize the Fock or IdTr basis matrices
        
        :param NB:          Dimensionality of the Hilbert space of the local 
                            oscillators considered (vacuum included!)
        :param basis_check: Indicize whether the 'Fock' or the 'IdTr' basis
                            is initialized
        :param vectorized:  Set to 'True' for initialize a vectorized basis
        :returns:           Representation of the Fock basis
    """ 
    basis    = [np.full((NB[q]*NB[q], NB[q], NB[q]),0.) for q in range(Q)]
    vecbasis = [np.full((NB[q]*NB[q],  NB[q]*NB[q]),0.) for q in range(Q)]    
    if bas == 'Fock':
        for q in range(Q):
            nb = 0
            for i in range(NB[q]):
                for j in range(NB[q]):
                    basis[q][nb, i, j] = 1.
                    nb += 1
    elif bas == 'IdTr':
        for q in range(Q):
            #First we identify the diagonal terms
            for i in range(NB[q]-1):
                basis[q][0, i, i]     = 1./np.sqrt(NB[q])
                basis[q][i + 1, i, i] = 1./np.sqrt(2.)
                basis[q][i + 1, i + 1, i + 1] = -1./np.sqrt(2.)
            basis[q][0, NB[q] - 1, NB[q] - 1] = 1./np.sqrt(NB[q])                
            #Now we move to the off diagonal terms
            nb = NB[q]
            for i in range(NB[q]):
                for j in range(i + 1,NB[q]):
                    basis[q][nb,i,j] = 1./np.sqrt(2.)
                    basis[q][nb,j,i] = 1./np.sqrt(2.)
                    nb += 1
                    basis[q][nb,i,j] = 1./np.sqrt(2.)
                    basis[q][nb,j,i] = -1./np.sqrt(2.)
                    nb += 1 
            #We orthonormalize thanks to the QR decomposition
            basis_resh = np.full((NB[q]*NB[q], NB[q]*NB[q]),0.)
            for i in range(NB[q]*NB[q]):
                basis_resh[:, i] = np.reshape(basis[q][i], (NB[q]*NB[q],))
            basis_q, basis_r = spla.qr(basis_resh)
            for i in range(NB[q]*NB[q]):
                basis[q][i] = np.reshape(basis_q[:, i], (NB[q],NB[q]))
                #The multiplication for the minus is not necessary, it is just for having
                #an identity matrix as first element (with positive coefficients)
                basis[q][i] /= -1.
            #Check whether the orthonormalization changes to diagonal element
            for i in range(NB[q]):
                if basis[q][0][i,i] <= (1/np.sqrt(NB[q]) - 10**(-10)) or basis[q][0][i,i] >= (1/np.sqrt(NB[q]) + 10**(-10)):
                    sys.exit('Orthonormalization of the basis changed diagonal element. Changes in the partial trace required')
    else:
        sys.exit('Error in basis parameter. Choose (Fock) or (IdTr)')  
    if vectorized is True:
        for q in range(Q):
            for i in range(NB[q]*NB[q]):
                vecbasis[q][i] = basis[q].copy()[i].reshape(NB[q]*NB[q])
        return basis, vecbasis
    else:
        return basis
    
def fock_to_idtr():
    """
        Initialize the transformation matrix between the two basis
        
        :param idtr_basis: identity+traceless matrices basis
        :param NB: dimensionality of the Hilbert space of the local oscillator
        :returns: transformation matrix between the fock and the traceless basis
    """
    focktoidtr = [np.full((NB[q]*NB[q], NB[q]*NB[q]),0.) for q in range(Q)]
    basisfock = basis_initialization(bas = 'Fock')
    basisidtr = basis_initialization(bas = 'IdTr')
    for q in range(Q):
        for i in range(NB[q]*NB[q]):
            for j in range(NB[q]*NB[q]):
                focktoidtr[q][i, j]=np.trace(np.conjugate(basisfock[q][i].T) @ basisidtr[q][j])   
    return focktoidtr

def single_to_double():
    """
        Defines the mapping between single to double index notation for the
        electronical degrees of freedom
    
        :param N: number of electronical sites
        :returns: map between single to double index notation for the
                  electronical degrees of freedom
    """
    std=np.full((N*N,2), 1)
    ll = 0
    for i in range(N):
        for j in range(N):
            std[ll]=[i, j]
            ll += 1
    return std

def double_to_single():
    """
        Defines the mapping between double to single index notation for the
        electronical degrees of freedom
    
        :param N: Number of electronical sites
        :returns: map between double to single index notation for the
                  electronical degrees of freedom
    """
    dts=np.full((N,N), 1)
    ll = 0
    for i in range(N):
        for j in range(N):
            dts[i, j] = ll
            ll += 1
    return dts