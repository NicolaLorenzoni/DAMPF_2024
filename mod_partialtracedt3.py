import ray
import numpy as np

#-- Dampf modules ------------------------------------------------------------
from mod_parametersdt3 import N, NB, Q
    
"""
    In this module we collect all the functions associated with
    the computation of reduced density matrices
"""
   
@ray.remote(num_returns = N*N)
def idtr_rhot_el(*rhott):
    """
        Compute the electronical reduced density matrix in the 
        Identity + Traceless matrices basis
        
        :param rhott:  Density matrix at a given time
    """
    
    res    = []
    factor = 1.
    for q in range(Q):
        factor = factor*(NB[q])**(N/2.)
    for j in range(N**2):
        #Here i deal with MPS for a given time and a given m,n couple
        prod = [1]
        for l in range(N*Q):
            prod = prod @ rhott[j].lt[l][:,0,:]
        res.append(*(factor*prod))
    return res

    """factor = 1.
    for q in range(Q):
        factor = factor*(NB[q])**(N/2.)
    res = [factor*mp.mparray.prune(rhott[j].get(tuple(0 for _ in range(N*Q)))).to_array() for j in range(N**2)]
    return res"""


@ray.remote(num_returns = N*Q)
def idtr_rhot_vibloc(basis, *rhott):
    """
        Compute the local oscillators reduced density matrix in the 
        Identity + Traceless matrices basis
        
        :param rhotvibt:  Reduced vibrational density matrix
        :param basis:     Identity + Traceless basis
    """
    res = []
    ind = 0
    for i in range(N):
        for q in range(Q):
            factor = 1.
            term   = np.zeros((NB[q],NB[q]), dtype = complex)
            for k in range(Q):
                if k == q:
                    factor = factor*(NB[k])**((N - 1)/2.)
                else:
                    factor = factor*(NB[k])**(N/2.)
            for j in range(NB[q]*NB[q]):
                sum_prod = 0
                for n in range(N):
                    prod = [1]
                    for l in range(N*Q):
                        if l != ind:
                            prod = prod @ rhott[N*n+n].lt[l][:,0,:]
                        else:
                            prod = prod @ rhott[N*n+n].lt[l][:,j,:]
                    sum_prod += prod
                term += sum_prod*factor*basis[q][j]
            res.append(term)
            ind += 1
    return res


@ray.remote(num_returns = N)
def idtr_rhot_vibloc_cond_exc(transf_matrix, basis, std, dts, *rhott):
    """
        Compute the local oscillators reduced density matrix conditional
        to exciton state in the Identity + Traceless matrices basis
        
        :transf_matrix:   Matrix that transform sites to excitons
        :param rhotvibt:  Reduced vibrational density matrix
        :param basis:     Identity + Traceless basis
    """
    res = [[] for exc in range(N)]
    for exc in range(N):
        ind = 0
        for i in range(N):
            for q in range(Q):
                res[exc].append(np.zeros((NB[q],NB[q]), dtype = complex))
                factor = 1.
                for k in range(Q):
                    if k == q:
                        factor = factor*(NB[k])**((N - 1)/2.)
                    else:
                        factor = factor*(NB[k])**(N/2.)
                for j in range(NB[q]*NB[q]):
                    sum_prod = 0
                    for m in range(N):
                        for n in range(N):
                            prod = [transf_matrix[exc, m]*np.conjugate(transf_matrix[exc, n])]
                            for l in range(N*Q):
                                if l != ind:
                                    prod = prod @ rhott[dts[m,n]].lt[l][:,0,:]
                                else:
                                    prod = prod @rhott[dts[m,n]].lt[l][:,j,:]
                            sum_prod += prod[0]
                    res[exc][ind] += factor*sum_prod*basis[q][j]
                ind += 1
    return res

                
                
                
                
                
                
                
                
                
                
    
    
    
