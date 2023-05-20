import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd


def get3D(X1, X2, K):
    """_summary_
    Find 3D coordinates on second image from matching points
    B = (# of mathced points)
    Args:
        X1 (np.ndarray): B * 2, Matched points (x, y)s in first image
        X2 (np.ndarray): B * 2, Matched points (x, y)s in second image
    Returns:
        X3D (np.ndarray): B * 3, 3D coordinates relative to second camera
    """
    print("map_initialization.py : Calculating an initial map...")
    c = np.array([K[0][2],K[1][2]])
    f = np.array([K[0][0],K[1][1]])
    N1 = (X1-c)/f
    N2 = (X2-c)/f
    B = X1.shape[0]
    R  = np.eye(3)
    t = np.array([[10], [0], [0]])
    P = np.concatenate((R, t), axis=1)
    
    A = np.zeros((B, 4, 4))
    
    #fill first 2 rows
    A[:, 0, 0] = 1
    A[:, 1, 1] = 1
    A[:, :2, 2] = -N2
    
    #fill last 2 rows
    p2 = np.tile(P[2], (B, 2, 1))
    p = np.tile(P[:2], (B, 1, 1))
    A[:, 2:, :] = p2*N1[:,:, None]-p
    
    U, S, VT = svd(A)
    
    X3D = VT[:,3, :3] / VT[:,3,3, None]
    print("map_initialization.py : All done!!! Map size =", X3D.shape)
    
    ### To verify the X3D, reconstruct 2D coordinates from 3D coordinates(Works quite well)
    ### To see the results, uncomment the following code
    
    # recons = X3D[:, :2]/X3D[:,[2]]
    # idx = recons[:,0].argsort()
    # recons1 = recons[idx,:]
    # X2_1 = X2[X2[:,0].argsort(), :]
    # print(np.concatenate((recons1, X2_1), axis=1))
    
    
    ### Drawing XYZ plot of X3D
    ### To see the results, uncomment the following code
    
    # fig = plt.figure(figsize=(10,10))
    # ax = fig.add_subplot(111, projection="3d")
    # ax.set_title("XYZ coordinates relative to second camera")
    # ax.set_xlabel("x")
    # ax.set_xlim(0, 50)
    # ax.set_ylabel("y")
    # ax.set_ylim(0, 40)
    # ax.set_zlabel("z")
    # ax.set_zlim(0, 0.02)
    # ax.scatter(X3D[:,0], X3D[:,1], X3D[:,2], marker='o', s=15)
    # plt.show()
    
    return X3D

