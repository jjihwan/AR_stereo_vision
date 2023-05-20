import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def mainPlane(X3D):
    """_summary_
    Find 3D dominant plane that corresponds to most 3D points by RANSAC
    B = (# of mathced points)
    Args:
        X3D (np.ndarray): B * 3, 3D coordinates relative to second camera
    Returns:
        dom_plane (np.ndarray): 1 * 4, (a, b, c, d) that represent the plane ax+by+cz+d = 0
    """
    print("map_initialization.py : Finding the dominat plane...")

    # manually select the iteration and threshold
    iteration = 300
    threshold = 3
    best_inlier = 0
    for ii in range(iteration):
        inlier = 0
        a = np.random.choice(range(np.shape(X3D)[0]),3,replace=False)
        u = X3D[a[0]]-X3D[a[1]]
        v = X3D[a[0]]-X3D[a[2]]
        normal = np.cross(u,v)
        normal = normal/np.linalg.norm(normal)
        d = -np.dot(normal,X3D[a[0]])
        for i in range(len(X3D)):
            distance = np.abs(np.dot(normal,X3D[i]+d))
            if distance < threshold:
                inlier += 1
        if inlier > best_inlier:
            print("dominant plane updated")
            best_inlier = inlier
            dom_plane = np.append(normal,d)  # [a,b,c,d] ; ax+by+cz+d = 0
    print("map_initialization.py : # of inlier points in dominant plane = ", best_inlier)

    return dom_plane

def plot_plane(dom_plane, X3D, K):
    ### Drawing XYZ plot of X3D
    ### To see the results, uncomment the following code

    xx, yy = np.meshgrid(range(-30,20),range(-20,20))
    zz = -(dom_plane[0]*xx+dom_plane[1]*yy+dom_plane[3])/dom_plane[2]
    
    # test 3D cuboid
    a = K[0][2]/K[0][0]
    b = K[1][2]/K[1][1]
    print(a)
    print(b)
    k = -dom_plane[3]/(-dom_plane[0]*a-dom_plane[1]*b+dom_plane[2])
    print(k)
    any2p = np.array([[-k*a,-k*b],[-k*a+5,-k*b]]) # (-k*a,-k*b,z) in the dom_plane is the point that projects to image point (0,0) 
    z = -(dom_plane[0]*any2p[:,0]+dom_plane[1]*any2p[:,1]+dom_plane[3])/dom_plane[2]
    print(z)
    any2p = np.concatenate((any2p, np.expand_dims(z,1)),1)
    u = any2p[0]-any2p[1]
    u = u/np.linalg.norm(u)
    v = np.cross(u,dom_plane[0:3])
    v = v/np.linalg.norm(v)
    x = [any2p[0]]
    for i in range(9):
        for j in range(14):
            if i==0 and j==0:
                continue
            xij = [any2p[0]+u*5*i+v*5*j]
            x = np.concatenate((x,xij))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("XYZ coordinates relative to second camera")
    ax.set_xlabel("x")
    ax.set_xlim(-20, 20)
    ax.set_ylabel("y")
    ax.set_ylim(-20, 20)
    ax.set_zlabel("z")
    ax.set_zlim(0, 50)
    ax.plot_surface(xx,yy,zz,alpha=0.2)
    ax.scatter(x[:,0],x[:,1],x[:,2],marker='.', s=10)
    ax.scatter(X3D[:,0], X3D[:,1], X3D[:,2], marker='o', s=15)
    plt.show()

    return x


def obj3Dto2D(keyImg, K, dom_plane, X3D, x):
    x = np.matrix(x)
    p = x/x[:,2]
    u = np.array(p[:,0]).flatten()
    v = np.array(p[:,1]).flatten()
    p = np.vstack((u, v)).T
    c = np.array([K[0][2],K[1][2]])
    f = np.array([K[0][0],K[1][1]])
    p = p*f+c
    plt.imshow(keyImg)
    plt.scatter(p[:,0], p[:,1],s=10)

    # plot feature
    X3D = np.matrix(X3D)
    X3D = X3D/X3D[:,2]
    a = np.array(X3D[:,0]).flatten()
    b = np.array(X3D[:,1]).flatten()
    
    pfeat = np.vstack((a, b)).T
    pfeat = pfeat*f+c
    plt.scatter(pfeat[:,0],pfeat[:,1],s=10)
    plt.axis("off")
    plt.show()

def det_plane(X3D, img, K):
    dom_plane = mainPlane(X3D)
    x = plot_plane(dom_plane, X3D, K)
    obj3Dto2D(img, K, dom_plane, X3D, x)