
import h5py
import numpy as np
import pickle


N = 100
RES = 256

filename = 'l96_0.h5'
with h5py.File(filename, 'r') as f:
    a_group_key = list(f.keys())[0]
    data = list(f[a_group_key])
    TRAJ = np.asarray(data)


PROJ = []
for i in range(9):
    for j in range(9):
        PROJ.append([i, j])

TRAJ_2D = np.zeros((N,len(TRAJ),len(TRAJ[0]),2))
for i in range( 100 ):
     print(i)
     filename = 'l96_' + str(i) + '.h5'
     with h5py.File(filename, 'r') as f:
        a_group_key = list(f.keys())[0]
        data = list(f[a_group_key])
        TRAJ = np.asarray(data)
        TRAJ_2D[i,:,:,0] = TRAJ[:,:,2]
        TRAJ_2D[i,:,:,1] = TRAJ[:,:,3]


X_LIM = [-9.0, 11.0]
Y_LIM = [-6.0, 10.0]

X_EDGES = np.linspace(X_LIM[0], X_LIM[1], RES)
Y_EDGES = np.linspace(Y_LIM[0], Y_LIM[1], RES)

H = []
for j in range(len(TRAJ[0])):
    X=np.ravel(TRAJ_2D[:,:,j,0])
    Y=np.ravel(TRAJ_2D[:,:,j,1])
    REMOVE = np.where(np.isnan(X)|np.isnan(Y))
    X = np.delete(X,REMOVE)
    Y = np.delete(Y,REMOVE)
    HIST = np.histogram2d(X,Y,bins=(X_EDGES, Y_EDGES))
    H.append(HIST)


with open('H'+str(2)+str(3)+'.p','wb') as g:
    pickle.dump(H,g)
