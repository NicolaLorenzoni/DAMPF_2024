import numpy        as np
import scipy.linalg as spla
import mpnum_v2        as mp

#-- Dampf modules ------------------------------------------------------------
from mod_structuresdt3 import MPOLIST
from mod_utilsdt3      import Boltzmann_aver_num
from mod_parametersdt3 import N, Q, NB, Omega_osc,\
                              J, HR, drate, dt, Omega_el

"""
    In this module we compute the evolution operators starting from the model
    of the Hamiltonian. In here, the density matrix is vectorize so that the
    dissipator is expressed as an operator (not a superoperator).
    Since many operators act in an equal way on the electronical sites, we 
    only need to define smaller Q dimensional structures to build them.
"""


def reduce_to_target_size(reduction_matrix, matrix_to_reduce):
    """
        :param reduction_matrix: Matrix with zeros on rows/cols to remove
        :param matrix_to_reduce: Matrix to cut
    """
    # Identify zero rows and columns in M
    zero_rows = np.where(~reduction_matrix.any(axis=1))[0]
    zero_columns = np.where(~reduction_matrix.any(axis=0))[0]
    
    # Eliminate corresponding rows and columns in A
    matrix_reduced = np.delete(matrix_to_reduce, zero_rows, axis=0)  # Remove rows
    matrix_reduced = np.delete(matrix_reduced, zero_columns, axis=1)  # Remove columns
    
    return matrix_reduced


def model_initialization(basis, vecbasis, dts):
    """
        :param basis:      Basis considered
        :param vecbasis:   Vectorized form of the basis
        :param dts:        Double to single index mapping
    """

    mult = 5

    #Creation (aqc), Annihilation (aqa)
    aqc = [np.diag(np.sqrt(np.arange(1, mult*NB[q], dtype=float)), k=-1) for q in range(Q)]
    aqa = [np.diag(np.sqrt(np.arange(1, mult*NB[q], dtype=float)), k=+1) for q in range(Q)]  
    idosc = [np.identity(mult*NB[q]) for q in range(Q)]    

    #Electronical Hamiltonian
    He =   np.diag(Omega_el, k = 0) + J

    #Vibrational Hamiltonian 
    Hv  = [(Omega_osc[q]) * (aqc[q] @ aqa[q]) for q in range(Q)]   

    #Coupling Hamiltonian
    Hev = [(Omega_osc[q]) * np.sqrt(HR[q]) * (aqa[q] + aqc[q]) for q in range(Q)]   

    #Dissipation operator
    D = [drate[q] * (Boltzmann_aver_num(q) + 1.)* (np.kron(aqa[q], aqa[q]) - 0.5 * (np.kron(aqc[q] @ aqa[q], idosc[q])\
         + np.kron(idosc[q], aqc[q] @ aqa[q]))) + drate[q] * Boltzmann_aver_num(q)* (np.kron(aqc[q], aqc[q])\
         - 0.5 * (np.kron(aqa[q] @ aqc[q], idosc[q]) + np.kron(idosc[q], aqa[q] @ aqc[q]))) \
        for q in range(Q)]                                                                                

    #Computing evolution operators
    Ue  = spla.expm(-(1j)*He*dt)
    
    Uv_ext  = [spla.expm(-(1j)*Hv[q]*dt) for q in range(Q)]
    
    Uev_ext = [spla.expm(-(1j)*Hev[q]*dt/2.) for q in range(Q)]
    
    UD_ext =  [spla.expm((D[q] + np.full(D[q].shape, 0., dtype=complex))*dt/2.) for q in range(Q)]
    
    red1 = [np.zeros((mult*NB[q], mult*NB[q])) for q in range(Q)]
    for q in range(Q):
        for i in range(NB[q]):
            red1[q][i,i] = 1
    red2 = [np.kron(red1[q], red1[q]) for q in range(Q)]
    
    Uv = [reduce_to_target_size(red1[q], Uv_ext[q]) for q in range(Q)]
    
    Uev = [reduce_to_target_size(red1[q], Uev_ext[q]) for q in range(Q)]
    
    UD  = [reduce_to_target_size(red2[q], UD_ext[q]) for q in range(Q)]    
    #EVOLUTION AS MPO
    '''
    Since Hv and D1 do not contains a part acting on the electronic dof, we only need Q local 
    tensors in order to identify the MPO structure. Moreover, our collection of N*N MPOs is 
    nothing but the repetition of the same MPO for any (m,n) couple.
    '''
        
    ####################################################################    
    #Hv and D
    loctenUv = mp.factory.zero(sites = Q, ldim = ((NB[q]*NB[q], NB[q]*NB[q]) for q in range(Q)), rank = 1)
    loctenD  = mp.factory.zero(sites = Q, ldim = ((NB[q]*NB[q], NB[q]*NB[q]) for q in range(Q)), rank = 1)

    for q in range(Q):
        temptens1 = np.zeros((1, NB[q]*NB[q], NB[q]*NB[q], 1), dtype = complex)
        temptens2 = np.zeros((1, NB[q]*NB[q], NB[q]*NB[q], 1), dtype = complex)
        for i in range(NB[q]*NB[q]):
            for j in range(NB[q]*NB[q]):
                temptens1[0, i, j, 0] = np.trace(np.conjugate(basis[q][i].T) @ Uv[q] @ basis[q][j] @ np.conjugate(Uv[q].T))
                temptens2[0, i, j, 0] = np.transpose(vecbasis[q][i]) @ UD[q] @ vecbasis[q][j]    
        loctenUv.lt.update(q, temptens1)
        loctenD.lt.update(q, temptens2)

    UvMPO = mp.mparray.MPArray([loctenUv.lt[q] for q in range(Q)] * N)
    
    DMPO = mp.mparray.MPArray([loctenD.lt[q] for q in range(Q)] * N)

    '''
    Here we create a prototype of nested list with N*N elements. Each elements is
    a list containing the local tensors spanning an MPS/MPO
    '''
    totallist1 = []
    for i in range(N*N):
        totallist1.append([])
    '''
    Hev gives different terms. For (m,n=m) we have a certain local tensors acting on 
    the oscillators on the n site, for (m,n) with m different respect n we have two 
    local tensors, acting on the oscillators at the m and n sites. Thus we need to 
    compute three different local tensors. With them, we can define the N*N collection 
    of MPO in which at each couple (m,n) there are non identities terms only on one 
    or two electronic sites. Differently from the Hv,D1 cases now our collection is 
    no more given by the repetition of the same MPO over the (m,n) couples, but each 
    of this couples is associated to a different MPO with identities elements here 
    and there.
    '''

    loctenUev1 = mp.factory.zero(sites = Q, ldim=[(NB[q]*NB[q], NB[q]*NB[q]) for q in range(Q)], rank = 1)
    loctenUev2 = mp.factory.zero(sites = Q, ldim=[(NB[q]*NB[q], NB[q]*NB[q]) for q in range(Q)], rank = 1)
    loctenUev3 = mp.factory.zero(sites = Q, ldim=[(NB[q]*NB[q], NB[q]*NB[q]) for q in range(Q)], rank = 1)
    lociden = mp.factory.eye(sites=Q, ldim=[(NB[q]*NB[q]) for q in range(Q)])

    for q in range(Q):
        temptens1 = np.zeros((1, NB[q]*NB[q], NB[q]*NB[q], 1), dtype = complex)
        temptens2 = np.zeros((1, NB[q]*NB[q], NB[q]*NB[q], 1), dtype = complex)
        temptens3 = np.zeros((1, NB[q]*NB[q], NB[q]*NB[q], 1), dtype = complex)
        for i in range(NB[q]*NB[q]):
            for j in range(NB[q]*NB[q]):
                temptens1[0, i, j, 0] = np.trace(np.conjugate(basis[q][i].T) @ Uev[q] @ basis[q][j] @ np.conjugate(Uev[q].T))
                temptens2[0, i, j, 0] = np.trace(np.conjugate(basis[q][i].T) @ Uev[q] @ basis[q][j])
                temptens3[0, i, j, 0] = np.trace(np.conjugate(basis[q][i].T) @ basis[q][j] @ np.conjugate(Uev[q].T))
        loctenUev1.lt.update(q, temptens1)
        loctenUev2.lt.update(q, temptens2)
        loctenUev3.lt.update(q, temptens3)
    

    '''
    Since Hev acts non trivially only on determined electronical sites, we define 
    four sets of  Q local tensors: three for the three different ways in which Hev
    acts, one for the set in which Hev acts as an identity.
    '''

    sitelist1 = []
    sitelist2 = []
    sitelist3 = []
    siteiden  = []
    for q in range(Q):
        siteiden.append(lociden.lt[q])
        sitelist1.append(loctenUev1.lt[q])
        sitelist2.append(loctenUev2.lt[q])
        sitelist3.append(loctenUev3.lt[q])
    '''
    Now we chain with the identities so to recover the whole MPO
    '''
    chainlist1 = []
    bigger  = 0
    smaller = 0
    UevMPO  = MPOLIST(N, Q, NB).list
    for row in range(N):
        for col in range(N):
            #First step: i map the single electronical index back into the double (m,n) one
            l = dts[row, col]
            smaller = min(row, col)
            bigger  = max(row, col)
    
            #DIAGONAL TERMS
            #In this case indeces[0]=indeces[1] gives the evaluated electronical site
            if row == col:                     
                chainlist1 = []
                left  = []
                right = []
                if row == 0:
                    left = []
                else:
                    for i in range(row):
                        left += siteiden
                if row == N - 1:
                    right = []
                else:
                    for i in range(N - 1, row, -1):
                        right += siteiden
                chainlist1 = left + sitelist1 + right 
                totallist1[l] = chainlist1
            ###########################
    
            #OFF DIAGONAL TERMS
            else:
                chainlist1 = []
                left   = []
                middle = []
                right  = []
                if smaller == 0:
                    left = []
                else:
                    for i in range(smaller):
                        left += siteiden
                for i in range(smaller + 1, bigger, 1):
                    middle += siteiden
                if bigger == N - 1:
                    right = []
                else:
                    for i in range(N - 1, bigger, -1):
                        right += siteiden
                if row < col:
                    chainlist1 = left + sitelist2 + middle + sitelist3 + right    
                else:
                    chainlist1 = left + sitelist3 + middle + sitelist2 + right
                totallist1[l] = chainlist1
            ###########################
            UevMPO[l] = mp.mparray.MPArray(totallist1[l])
    
    step1MPO = MPOLIST(N, Q, NB).list
    UvUevDMPOp1 = MPOLIST(N, Q, NB).list
    UvUevDMPOp2 = MPOLIST(N, Q, NB).list
        
    for j in range(N*N):
        step1MPO[j] = mp.dot(UevMPO[j], DMPO, axes=(-1,0))   
    for j in range(N*N):
        UvUevDMPOp1[j] = mp.dot(UvMPO, step1MPO[j], axes=(-1,0))

    for j in range(N*N):
        UvUevDMPOp2[j] = mp.dot(DMPO, UevMPO[j], axes=(-1,0))           
        
    return [Ue, UvUevDMPOp1, UvUevDMPOp2]




          

