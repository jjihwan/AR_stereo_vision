import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def planeRANSAC(X3D, iteration, threshold):
    """_summary_
    Find 3D dominant plane that corresponds to most 3D points by RANSAC
    B = (# of mathced points)
    Args:
        X3D (np.ndarray): B * 3, 3D coordinates relative to second camera
        threshold (int) : max distance between inlier points and dominant plane
        iteration (int) : RANSAC iteration to find dominant plane
    Returns:
        dom_plane (np.ndarray): 1 * 4, (a, b, c, d) that represent the plane ax+by+cz+d = 0
    """
    print("plane.py : Finding the dominat plane...")

    best_inlier = 0
    for ii in range(iteration):
        inlier = 0
        randi = np.random.choice(range(np.shape(X3D)[0]), 3, replace=False)
        u = X3D[randi[0]]-X3D[randi[1]]
        v = X3D[randi[0]]-X3D[randi[2]]
        normal = np.cross(u, v)
        normal = normal/np.linalg.norm(normal)
        if normal[2] < 0:
            normal = -normal
        d = -np.dot(normal, X3D[randi[0]])
        for i in range(len(X3D)):
            distance = np.abs(np.dot(normal, X3D[i])+d)
            if distance < threshold:
                inlier += 1
        if inlier > best_inlier:
            print("plane.py : dominant plane updated",
                  '(iteration', ii, '/', iteration, ')')
            best_inlier = inlier
            dom_plane = np.append(normal, d)  # [a,b,c,d] ; ax+by+cz+d = 0
    print("plane.py : # of inlier points in dominant plane = ", best_inlier)

    return dom_plane


def plot_plane(dom_plane, X3D, K):
    """_summary_
    plot 3D plane with 3D points in world coordinate
    B = (# of mathced points)
    n = (# of grid points manually selected with wi and hi)
    Args:
        dom_plane (np.ndarray): 1 * 4, (a, b, c, d) that represent the plane ax+by+cz+d = 0
        X3D (np.ndarray): B * 3, 3D coordinates relative to second camera
        K (np.ndarray): 3 * 3, intrinsic matrix of camera
    Returns:
        planeGrid3D (np.ndarray): n * 3, 3D coordinates of vertically grid points in dominant plane
    """
    print("plane.py : 3D Plot the dominat plane...")

<<<<<<< HEAD
    
    meshx_plane, meshy_plane = np.meshgrid(range(-50,50),range(-50,20))
    meshz_plane = -(dom_plane[0]*meshx_plane+dom_plane[1]*meshy_plane+dom_plane[3])/dom_plane[2]
    
=======
    meshx_plane, meshy_plane = np.meshgrid(range(-30, 20), range(-20, 20))
    meshz_plane = -(dom_plane[0]*meshx_plane+dom_plane[1]
                    * meshy_plane+dom_plane[3])/dom_plane[2]

>>>>>>> d99a1324a6e93970d1932cae412a49c514f72f36
    # test 3D cuboid
    u0_norImg = -K[0][2]/K[0][0]  # -cx/fx
    v0_norImg = -K[1][2]/K[1][1]  # -cy/fy
    # constant multiple to project to dom_plane
    k = -dom_plane[3]/(dom_plane[0]*u0_norImg+dom_plane[1]
                       * v0_norImg+dom_plane[2])
    # (-k*a,-k*b,z) in the dom_plane is the point that projects to image point (0,0)
    init2pxy = np.array([[k*u0_norImg, k*v0_norImg],
                        [k*u0_norImg, k*v0_norImg-5]])
    init2pz = -(dom_plane[0]*init2pxy[:, 0]+dom_plane[1]
                * init2pxy[:, 1]+dom_plane[3])/dom_plane[2]
    init2p = np.concatenate((init2pxy, np.expand_dims(init2pz, 1)), 1)
    u_grid = init2p[0]-init2p[1]
    u_grid = u_grid/np.linalg.norm(u_grid)
    v_grid = np.cross(u_grid, dom_plane[0:3])
    v_grid = v_grid/np.linalg.norm(v_grid)
    planeGrid3D = [init2p[0]]
<<<<<<< HEAD
    h = 14
    w = 18
    for hi in range(h): # 14
        for wi in range(w): # 18
            if wi==0 and hi==0:
                continue # init2p[0] is already added to palne3Dgrid
            gridij = [init2p[0]+u_grid*5*hi+v_grid*5*wi]
            planeGrid3D = np.concatenate((planeGrid3D,gridij))
=======
    for hi in range(13):
        for wi in range(18):
            if wi == 0 and hi == 0:
                continue  # init2p[0] is already added to palne3Dgrid
            gridij = [init2p[0]+u_grid*5*hi+v_grid*5*wi]
            planeGrid3D = np.concatenate((planeGrid3D, gridij))

>>>>>>> d99a1324a6e93970d1932cae412a49c514f72f36
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("XYZ coordinates relative to second camera")
    ax.set_xlabel("x")
    ax.set_xlim(-40, 40)
    ax.set_ylabel("y")
    ax.set_ylim(-40, 20)
    ax.set_zlabel("z")
<<<<<<< HEAD
    ax.set_zlim(0, 100)
    ax.plot_surface(meshx_plane,meshy_plane,meshz_plane,alpha=0.2)
    # ax.scatter(planeGrid3D[:,0],planeGrid3D[:,1],planeGrid3D[:,2],marker='.', s=10)
    for i in range(h):
        ax.plot([planeGrid3D[w*i][0],planeGrid3D[w*(i+1)-1][0]],[planeGrid3D[w*i][1],planeGrid3D[w*(i+1)-1][1]],zs=[planeGrid3D[w*i][2],planeGrid3D[w*(i+1)-1][2]],color='0.5', linewidth=1)
    for j in range(w):
        ax.plot([planeGrid3D[j][0],planeGrid3D[(h-1)*w+j][0]],[planeGrid3D[j][1],planeGrid3D[(h-1)*w+j][1]],zs=[planeGrid3D[j][2],planeGrid3D[(h-1)*w+j][2]],color='0.5', linewidth=1)
    ax.scatter(X3D[:,0], X3D[:,1], X3D[:,2], marker='o', s=15)
=======
    ax.set_zlim(0, 50)
    ax.plot_surface(meshx_plane, meshy_plane, meshz_plane, alpha=0.2)
    ax.scatter(planeGrid3D[:, 0], planeGrid3D[:, 1],
               planeGrid3D[:, 2], marker='.', s=10)
    ax.scatter(X3D[:, 0], X3D[:, 1], X3D[:, 2], marker='o', s=15)
>>>>>>> d99a1324a6e93970d1932cae412a49c514f72f36
    plt.show()

    planeGrid3D = planeGrid3D[:,np.newaxis]
    planeGrid3D = np.reshape(planeGrid3D, (h,w,3))

<<<<<<< HEAD
    return planeGrid3D

=======
def plane3Dto2D(keyImg, K, X3D, planeGrid3D):
    """_summary_
    plot 3D plane with 3D points in world coordinate
    B = (# of mathced points)
    n = (# of grid points manually selected with wi and hi)
    Args:
        keyImg: image that represent the world coordinate
        K (np.ndarray): 3 * 3, intrinsic matrix of camera
        X3D (np.ndarray): B * 3, 3D coordinates relative to second camera
        planeGrid3D (np.ndarray): n * 3, 3D coordinates of vertically grid points in dominant plane
    Returns:
        None
    """
    print("plane.py : Plot the dominant plane grid in 2D image ...")

    c = np.array([K[0][2], K[1][2]])
    f = np.array([K[0][0], K[1][1]])

    # plot dominant plane grid
    planeGrid3D = np.matrix(planeGrid3D)
    planeGrid_Img = planeGrid3D/planeGrid3D[:, 2]
    planeGrid_norImgxy = np.array(planeGrid_Img[:, 0:2])
    planeGrid_Img = planeGrid_norImgxy*f+c

    plt.imshow(keyImg)
    plt.scatter(planeGrid_Img[:, 0], planeGrid_Img[:, 1], s=10)

    # plot feature
    X3D = np.matrix(X3D)
    X3D = X3D/X3D[:, 2]
    XnorImg = np.array(X3D[:, 0:2])
    XImg = XnorImg*f+c

    plt.scatter(XImg[:, 0], XImg[:, 1], s=10)
    plt.axis("off")
    plt.show()
>>>>>>> d99a1324a6e93970d1932cae412a49c514f72f36

# def detect_plane(X3D, keyimg, K):
#     dom_plane = planeRANSAC(X3D, 100, 0.05) # X3D, iteration, threshold
#     planeGrid3D = plot_plane(dom_plane, X3D, K)
<<<<<<< HEAD
#     plot3Dto2D(keyimg, K, X3D, planeGrid3D)
=======
#     plane3Dto2D(keyimg, K, X3D, planeGrid3D)


def get_dominant_plane(M, F0, K):
    M.normal_vector = planeRANSAC(M.X_3D_0, 100, 0.05)
    planeGrid3D = plot_plane(M.normal_vector, M.X_3D_0, K)
    plane3Dto2D(F0, K, M.X_3D_0, planeGrid3D)
    return M
>>>>>>> d99a1324a6e93970d1932cae412a49c514f72f36
