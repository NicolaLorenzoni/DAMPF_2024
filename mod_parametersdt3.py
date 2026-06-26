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

from mpnum_v2 import utils
from functools import partial
import numpy as np


"""
    In here we fix all the parameters needed for the simulation
"""

#-- Conversion  ---------------------------------------------------------------
cm_to_fs      = 0.00018836515661233488         #fs to cm-1
eV_to_inv_cm  = 8065.543730730113
Kb            = 8.617333262145e-5*eV_to_inv_cm #The first value is in eV/K

#-- Editable parameters ------------------------------------------------------
BD       = 27                 # Cutoff in Bond dimension
err      = 10**(-8)           # Relative error admitted in compression before BD is reached
epsilon  = 10**(-8)           # Treshold of the considered coefficients constructed from Ue
time     = 1000.0*cm_to_fs    # Simulated time (number in fs)
dt       = 1.0*cm_to_fs       # Timestep for the evolution
dtdata   = 1.0*dt             # Timestep between data collection
backuptime  = 540000            # Seconds between data storing
basis_check = 'IdTr'          # Fixes the basis (Fock) or (IdTr)
vib_data = False               # Set to True if vibrational info is needed


#-- Electronical specs -------------------------------------------------------
N = 7
Omega_el = [1410., 1530., 1210., 1320., 1480., 1630., 1440.]
negative_shift = 11000.

#-- Electronical couplings ---------------------------------------------------
J_list = [[  0.0, -87.7,   5.5,  -5.9,   6.7, -13.7,  -9.9],\
          [-87.7,   0.0,  30.8,   8.2,   0.7,  11.8,   4.3],\
          [  5.5,  30.8,   0.0, -53.5,  -2.2,  -9.6,   6.0],\
          [ -5.9,   8.2, -53.5,   0.0, -70.7, -17.0, -63.3],\
          [  6.7,   0.7,  -2.2, -70.7,   0.0,  81.1,  -1.3],\
          [ -13.7, 11.8,  -9.6, -17.0,  81.1,   0.0,  39.7],\
          [ -9.9,   4.3,   6.0, -63.3,  -1.3,  39.7,   0.0]]
    
sum_HR = 0
f= open('BChlFMO4.5K.txt','r')
for y in f.readlines()[1:]:
    sum_HR += float(y.split()[1])
    

J = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        J[i,j] = J_list[i][j]
        
He = np.diag(Omega_el, k = 0) + J
#Columns of eigmat_el are the eigenvectors in the site basis
eig_el, transf_matrix = np.linalg.eig(He)


L = np.full((N, 3), 0.)
L[0][0] = -0.7410
L[0][1] = -0.5606
L[0][2] = -0.3696

L[1][0] = -0.8571
L[1][1] =  0.5038
L[1][2] = -0.1073

L[2][0] = -0.1971
L[2][1] =  0.9574
L[2][2] = -0.2110

L[3][0] = -0.7992
L[3][1] = -0.5336
L[3][2] = -0.2766

L[4][0] = -0.7396
L[4][1] =  0.6558
L[4][2] =  0.1641

L[5][0] = -0.1350
L[5][1] = -0.8792
L[5][2] =  0.4569

L[6][0] = -0.4951
L[6][1] = -0.7083
L[6][2] = -0.5031


#-- Local oscillators specs --------------------------------------------------
Omega_osc  = []
HR    = []
rate  = []
KbT   = []

f= open('Full_FMO_77K_1ps_31L_DAMPF_1ps.txt','r')
for y in f.readlines()[1:]:
    Omega_osc.append(float(y.split()[0]))
    rate.append(1.0/float(y.split()[1]))
    HR.append(float(y.split()[2]))
    KbT.append(Kb*float(y.split()[3]))
Q = len(Omega_osc)
drate = [1.0/rate[i] for i in range(Q)]

NB = [10 for q in range(4)] + [4 for q in range(4, Q)]
#NB = [14 for q in range(4)] + [6 for q in range(4, Q)]
#-----------------------------------------------------------------------------  

#-- Compression settings -----------------------------------------------------
compr_settings = dict(method='svd', relerr=err, rank=BD)

    
