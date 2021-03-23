"""PART 1: Filter U,W velocity field via POD method. -> PART 2: Integrate particle paths via Runge-Kutta 4"""
from functools import partial
from multiprocessing import Pool
import time
import math
import gc
import numpy as np

from scipy.interpolate import interp1d
import scipy.linalg as la
import scipy.interpolate
from numba import jit
import h5py

@jit(nopython=True)
def drdt(R,F,D):
    u=np.zeros(9)
    for i in range(9):
        u[i]=(-R[i]+R[i-1]*(R[i+1]-R[i-2])+F)*D
    return u

@jit(nopython=True)
def total(R):
    """summation step"""
    R[0, :] += (R[1, :]+R[4, :]+2.0*(R[2, :]+R[3, :]))/6.0
    return R[0, :]



def rk4(R, F, T, D, W):
    dr=np.zeros((5, 9))
    dr_mid=np.zeros(9)
    dr[0, :] = R
    dt = D
    traj = dr[0, :]
    for n in range(T-1):
        t = 2*n
        dr[1, :] = drdt(dr[0,:],F,dt)+dr[0,:]*W[t,:]
        dr_mid=dr[0, :]+0.5*dr[1, :]

        t += 1

        dr[2, :] = drdt(dr_mid, F, dt)+dr_mid*W[t,:]
        dr_mid=dr[0, :]+0.5*dr[2, :]
 
        dr[3, :] = drdt(dr_mid, F, dt)+dr_mid*W[t,:]
        dr_mid=dr[0, :]+dr[3, :]

        t += 1
        dr[4, :] = drdt(dr_mid, F, dt)+dr_mid*W[t,:]
        dr[0, :] = total(dr)

        traj = np.vstack((traj, dr[0, :]))

    #return traj[]
    return traj


if __name__ == '__main__':
    gc.enable()

    dat1 = np.loadtxt('setOf3_1.txt')
    dat2 = np.loadtxt('setOf3_1.txt')
    dat3 = np.loadtxt('setOf3_1.txt')

   
    

    print("Integrating trajectories")
    #flow=np.zeros((N_TRAJ,TAU,2))
    F_0 = 5.0
    N_TRAJ = 10000
    INIT = np.zeros((N_TRAJ,9))
    N_DAT = 1000
    TAU = 5*N_DAT
    DELTA_T = 0.001
    SIGMA = 0.02

    DAT=np.zeros((N_DAT,9))
    DAT[:,:3] = dat1[-N_DAT:,:3]-dat1[-(N_DAT+1):-1,:3]
    DAT[:,3:6] =  dat2[-N_DAT:,:3]-dat2[-(N_DAT+1):-1,:3]
    DAT[:,6:9] =  dat3[-N_DAT:,:3]-dat3[-(N_DAT+1):-1,:3]

    T_LORES=np.linspace(0,DELTA_T*TAU,N_DAT)
    T_HIRES=np.linspace(0,DELTA_T*TAU,2*TAU)
    DAT_I=np.zeros((2*TAU,9))

    for i in range(9):
        LIN_D = interp1d(T_LORES, DAT[:,i], kind='linear')
        DAT_I[:,i] = SIGMA*math.sqrt(DELTA_T)*LIN_D(T_HIRES)
  

    print(np.amax(DAT_I)) 
    C_SIZE = int(N_TRAJ/12.0)
    for j in range(100):
        np.random.seed()
        for i in range(N_TRAJ):
            INIT[i, :] = 0.5*np.random.randn(9) + F_0

        T_START = time.time()
        
        with Pool(processes=12) as p:
            RESULTS = p.map(partial(rk4, F=F_0, T=TAU, D=DELTA_T, W=DAT_I), INIT, chunksize=C_SIZE)
        TRAJ = np.asarray(RESULTS)[:,::10,:]
        h5f = h5py.File('l96_' + str(j) + '.h5', 'w')
        h5f.create_dataset('trajectories', data=np.float32(TRAJ))
        h5f.close()

        T_END = time.time()
        print(T_END - T_START)
