import numpy as np

#-- Dampf modules ------------------------------------------------------------
from mod_parametersdt3 import N, Q, vib_data, BD, NB, dt, cm_to_fs
 
"""
    In this module we collect the functions for storing in txt files and 
    plotting the collected results
"""



def data_storing(data, dts):
    """
        :param rhotel:       Reduced electronical density matrix
        :param rhotvibloc:   Reduced oscillator density matrix
        :param dts:          Double to single index mapping
        :param plot:         Set to True for generating plots
    """
    rhotel = data[0]
    data_points = len(rhotel)
    
    #-- Electronical data ----------------------------------------------------
    rhotel = np.array([np.array(rhotel[i]).reshape((N,N)) for i in range(data_points)])
    np.save('./output_data/rhotel_BD_'+str(BD)+'_dt_'+str(dt/cm_to_fs)+'_NB_'+str(NB[0])+'.npy', rhotel)

   
    
    #-- Vibrational data -----------------------------------------------------
    if vib_data == True:       
        rhotvibloc = data[1]
        rhotvibloc_exc = data[2]
        for ind in range(N*Q):
            np.save('./output_data/vib_data/rhotvib_ind_'+str(ind)+'.npy', np.array([rhotvibloc[i][ind] for i in range(data_points)])) 
            for exc in range(N):
                np.save('./output_data/vib_data/rhotvib_exc_'+str(exc)+'_ind_'+str(ind)+'.npy', np.array([rhotvibloc_exc[i][exc][ind] for i in range(data_points)])) 
