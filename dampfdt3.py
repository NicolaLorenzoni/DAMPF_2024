#-----------------------------------------------------------------------------#
#\   ________         _____         ___      ___    _______     _________    /#
#\  |  _____ \       / ___ \       |   \    /   |  |  ____ \   |  _______|   /#
#\  | |     \ \     / /   \ \      | |\ \  / /| |  | |    \ \  | |____       /#
#\  | |     | |    / /_____\ \     | | \ \/ / | |  | |____/ /  |  ____|      /#
#\  | |     | |   / _________ \    | |  \__/  | |  |  _____/   | |           /#
#\  | |____/ /   / /         \ \   | |        | |  | |         | |           /#
#\  |_______/   /_/           \_\  |_|        |_|  |_|         |_|           /#
#\                                                                           /#
#\                                                                           /#
#\      _________________Ulm university 2020_________________                /#
#\                                                                           /#
#-----------------------------------------------------------------------------#
"""
This version of DAMPF is different from the standard only for the quantity computed,
where we compute the reduced density matrices of the oscillators conditional to
exciton state
"""

#-- External modules --------------------------------------------------------
import ray
import sys
import numpy               as np
import mpnum_v2            as mp
import time                as timelib
import os
import h5py

#-- Dampf modules -----------------------------------------------------------
from mod_parametersdt3    import N , Q, NB, time, dt, dtdata, basis_check, backuptime,\
                                 vib_data, eig_el, transf_matrix, L, BD, cm_to_fs
from mod_partialtracedt3  import idtr_rhot_el, idtr_rhot_vibloc, idtr_rhot_vibloc_cond_exc
from mod_basisdt3         import basis_initialization, fock_to_idtr, single_to_double,\
                                 double_to_single
from mod_modeldt3         import model_initialization
from mod_initialstatedt3  import state_initialization
from mod_utilsdt3         import Uerelevant, adjoint_mapping
from mod_evolutiondt3     import evolve
from mod_structuresdt3    import UNCMPSLIST
from mod_datastoringdt3   import data_storing


np.set_printoptions(precision=2)

#-- Non editable parameters --------------------------------------------------
ntime      = (int)(time/dt)         # Iterations of timestep evolution
ntimedata  = (int)(time/dtdata)     # Iterations of data collection
    
    
#-- Readin parameters passed from submit batch script ------------------------
num_cpus_per_node= int(sys.argv[1])
num_nodes= int(sys.argv[2])
ray_mem_per_node= int(sys.argv[3])

print(num_cpus_per_node)
print(num_nodes)
print(ray_mem_per_node)
#-----------------------------------------------------------------------------
    
#-- Ray initialization -------------------------------------------------------
ray.init(address='auto', _node_ip_address=os.environ["ip_head"].split(":")[0], _redis_password=os.environ["redis_password"])

#-- Saving starting time -----------------------------------------------------
start = timelib.perf_counter()

#-- Basis and indeces initialization -----------------------------------------
basislist  = basis_initialization(bas = basis_check, vectorized = True)
focktoidtr = fock_to_idtr()
std        = single_to_double()
dts        = double_to_single()
basis      = ray.put(basislist[0])
vecbasis   = basislist[1]

#-- Importing evolution operators --------------------------------------------
operlist   = model_initialization(ray.get(basis),vecbasis,dts)
Ue         = operlist[0]
UvUevDMPO  = [[ray.put(operlist[i][j]) for j in range(N*N)] for i in range(1,3)]

#-- Importing mapping to the relevant elements -------------------------------
relev      = Uerelevant(Ue, std)
Uerel_ind  = relev[0]
coeff      = ray.put(relev[1])

#-- Importing mapping to adjoint density matrix for IdTr basis ---------------
adjmpo = ray.put(adjoint_mapping()) 

@ray.remote(num_returns = N**2)
def normalization(*rhot_rhotel_t):
    res=[]
    norm = 0.
    for j in range(N):
        norm += rhot_rhotel_t[N**2 + N*j + j]
    for j in range(N*N):
        res.append(rhot_rhotel_t[j]/norm)
    return res 


light_polariz = [0.25, 0.85, 0.0]
norm_light_pol = 0.
for x in range(3):
    norm_light_pol += light_polariz[x]**2
light_polariz /= np.sqrt(norm_light_pol)

#-- Initializing time zero state as excited via polarized pulse  -------------
rho0    = UNCMPSLIST(N, Q, NB).list
rho0_norm = 0.
for i in range(N):
    rho0_norm += np.abs(np.dot(L[i], light_polariz))**2
state_vib_0 = state_initialization(focktoidtr)
for i in range(N):
    for j in range(N):
        rho0[dts[i,j]] = state_vib_0*np.dot(L[i], light_polariz)*np.dot(L[j], light_polariz)/np.sqrt(rho0_norm)



#-- Initialize structure for data collection ---------------------------------
rhotel     = []
if vib_data == True:
    rhotvibloc = []
    rhotvibloc_exc = []

#-- Evolution ----------------------------------------------------------------
state = [ray.put(rho0[j]) for j in range(N*N)]
#state = [mp.MPArray.load('state_'+str(j)+'_BD_'+str(BD)+'_dt_'+str(dt/cm_to_fs)+'.hdf5') for j in range(N*N)]
#state = [ray.put(state[j]) for j in range(N*N)]
time_check = timelib.perf_counter()
endsingle = 0.
span = 5

if basis_check=='IdTr':
    for td in range(span):
        print('Progress: %.2f'%(td/ntimedata*100.)+'%')
        #Normalization
        rhotel.append(idtr_rhot_el.remote(*state))
        state      = normalization.remote(*state, *rhotel[td])
        
        startsingle = timelib.perf_counter()   
        state       = evolve(state, UvUevDMPO, coeff, Uerel_ind, adjmpo, dts) 
        endsingle   = timelib.perf_counter()
        print('time per step: {}'.format(endsingle-startsingle))
            
        #Data collection
        if vib_data == True:
            rhotvibloc.append(idtr_rhot_vibloc.remote(basis, *state))
            rhotvibloc_exc.append(idtr_rhot_vibloc_cond_exc.remote(transf_matrix, basis, std, dts,  *state))
        #Data storage and eviction from object memory
        rhotel[td] = ray.get(rhotel[td])
    for td in range(span, ntimedata +1):
        print('Progress: %.2f'%(td/ntimedata*100.)+'%')
        #Normalization
        rhotel.append(idtr_rhot_el.remote(*state))
        state      = normalization.remote(*state, *rhotel[td])
        
        startsingle = timelib.perf_counter()   
        state       = evolve(state, UvUevDMPO, coeff, Uerel_ind, adjmpo, dts) 
        endsingle   = timelib.perf_counter()
        print('time per step: {}'.format(endsingle-startsingle))
            
        #Data collection
        if vib_data == True:
            rhotvibloc.append(idtr_rhot_vibloc.remote(basis, *state))
            rhotvibloc_exc.append(idtr_rhot_vibloc_cond_exc.remote(transf_matrix, basis, std, dts, *state))
            rhotvibloc[td - span] = ray.get(rhotvibloc[td - span])
            rhotvibloc_exc[td - span] = ray.get(rhotvibloc_exc[td - span])
        #Data storage and eviction from object memory
        rhotel[td] = ray.get(rhotel[td])      
        #Backup every backuptime seconds
        if (endsingle - time_check) >= backuptime:
            if vib_data == True:
                for i in range(td - span + 1, td + 1):
                    rhotvibloc[i] = ray.get(rhotvibloc[i])
                    rhotvibloc_exc[i] = ray.get(rhotvibloc_exc[i])
                data_storing([rhotel, rhotvibloc, rhotvibloc_exc], dts)
                for i in range(td - span + 1, td + 1):
                    rhotvibloc[i] = ray.put(rhotvibloc[i])
                    rhotvibloc_exc[i] = ray.put(rhotvibloc_exc[i]) 
            else:
                data_storing([rhotel], dts)
            state = [ray.get(state[j]) for j in range(N*N)]
            for i in range(N*N):
                state[i].dump('state_'+str(i)+'_BD_'+str(BD)+'_dt_'+str(dt/cm_to_fs)+'.hdf5')
            state = [mp.MPArray.load('state_'+str(i)+'_BD_'+str(BD)+'_dt_'+str(dt/cm_to_fs)+'.hdf5') for i in range(N*N)]
            state = [ray.put(state[j]) for j in range(N*N)]
            time_check = endsingle         
else:
    sys.exit('Uncorret value of basis_check')
    
if vib_data == True:
    for i in range(td - span + 1, len(rhotvibloc)):
        rhotvibloc[i] = ray.get(rhotvibloc[i])
        rhotvibloc_exc[i] = ray.get(rhotvibloc_exc[i])
    data_storing([rhotel, rhotvibloc, rhotvibloc_exc], dts) 
else:
    data_storing([rhotel], dts)   

end=timelib.perf_counter()
print('Time taken ',end-start)
   
ray.shutdown()
