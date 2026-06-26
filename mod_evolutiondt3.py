import ray
from mpnum_v2 import dot

#-- Dampf modules ------------------------------------------------------------
from mod_parametersdt3 import N, dt, dtdata, compr_settings

"""
    Within this module we collect the functions generating the time
    evolution of the system
"""

#-- Local evolution remote functions -----------------------------------------
@ray.remote
def localupdate_UvUevDMPO_diag(MPO, rhott):
    """
        Evolve the density matrices according to a local operator
        
        :param MPO:     Local operator
        :param rhott:   Density matrix at a fixed time   
    """
    return dot(MPO, rhott, axes=(-1,0))

@ray.remote(num_returns = 2)
def localupdate_UvUevDMPO_offdiag(MPO, rhott, adjmpo):
    """
        Evolve the density matrices according to a local operator
        
        :param MPO:     Local operator
        :param rhott:   Density matrix at a fixed time   
    """
    res = [dot(MPO, rhott, axes=(-1,0))]
    res.append(dot(adjmpo, res[0], axes = (-1, 0)).conj())
    return res

#-- Non local evolution remote functions -------------------------------------
@ray.remote
def el_sum_diag(coeffterm, el, *relrhott):
    """
        Evolve the density matrices according to the non local operator Ue
        
        :param coeffterm:    Relvant coefficients associated with the 
                             electronical index el
        :param el:           Electronical index labelling the considered
                             MPS
        :param relrhott:     Density matrix at a fixed time
    """
    temp = coeffterm[el][0]*relrhott[0]
    for l in range(1,len(relrhott)):
        temp += coeffterm[el][l]*relrhott[l]
        temp.compress(**compr_settings)
    return temp

@ray.remote(num_returns = 2)
def el_sum_offdiag(coeffterm, el, adjmpo, *relrhott):
    """
        Evolve the density matrices according to the non local operator Ue
        
        :param coeffterm:    Relvant coefficients associated with the 
                             electronical index el
        :param el:           Electronical index labelling the considered
                             MPS
        :param relrhott:     Density matrix at a fixed time
    """
    temp = coeffterm[el][0]*relrhott[0]
    for l in range(1,len(relrhott)):
        temp += coeffterm[el][l]*relrhott[l]
        temp.compress(**compr_settings)
    res = [temp, dot(adjmpo, temp, axes = (-1, 0)).conj()]
    return res
    

def evolve(state, UvUevDMPO, coeff, Uerel_ind, adjmpo, dts):
    """
        Evolve the system for the time between two successive collections
        of data
        
        :param tempa:       Density matrix at a given time
        :param UvUevDMPO:   List of local evolution operators
        :param coeff:       List of relevant coefficients for the non local
                            evolution
        :param Uerel_ind:   Array of relevant indeces of the Ue operator
    """    
    statecopy = state
    for c in range(int(dtdata/dt)):
        # First evolution according to the local operator U(Hv,Hev,D)
        # Diagonal elements
        for i in range(N):
            statecopy[dts[i,i]] = localupdate_UvUevDMPO_diag.remote(UvUevDMPO[0][dts[i,i]], statecopy[dts[i,i]])
        # Off diagonal elements
        for i in range(N - 1):
            for j in range(i + 1, N):           
                statecopy[dts[i,j]], statecopy[dts[j,i]] = localupdate_UvUevDMPO_offdiag.remote(UvUevDMPO[0][dts[i,j]], statecopy[dts[i,j]], adjmpo)
         
                
        # Evolution according to Ue 
        # Diagonal elements
        resd = []
        for i in range(N):  
            resd.append(el_sum_diag.remote(coeff, dts[i,i], *[statecopy[k] for k in Uerel_ind[dts[i,i]]]))
        # Off diagonal elements
        resod = []
        for i in range(N - 1):
            for j in range(i + 1, N):   
                resod.append(el_sum_offdiag.remote(coeff, dts[i,j], adjmpo, *[statecopy[k] for k in Uerel_ind[dts[i,j]]]))
                
        # Storing result (need to be done after computing all of them)
        for i in range(N):
            statecopy[dts[i,i]] = resd[i]
        idx = 0
        for i in range(N - 1):
            for j in range(i + 1, N):
                statecopy[dts[i,j]] = resod[idx][0]
                statecopy[dts[j,i]] = resod[idx][1]
                idx += 1
                
    
        #Second evolution according to the local operator U(Hv,Hev,D)
        # Diagonal elements
        for i in range(N):
            statecopy[dts[i,i]] = localupdate_UvUevDMPO_diag.remote(UvUevDMPO[1][dts[i,i]], statecopy[dts[i,i]])
        # Off diagonal elements
        for i in range(N - 1):
            for j in range(i + 1, N):           
                statecopy[dts[i,j]], statecopy[dts[j,i]] = localupdate_UvUevDMPO_offdiag.remote(UvUevDMPO[1][dts[i,j]], statecopy[dts[i,j]], adjmpo)
                    
    ray.wait(state, num_returns = N*N)
    return statecopy