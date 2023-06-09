import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def planeRANSAC(X3Dref, iteration, threshold):
    """_summary_
    Find 3D dominant plane that corresponds to most 3D points by RANSAC
    B = (# of mathced points)
    Args:
        X3Dref (np.ndarray): B * 3, 3D coordinates relative to second camera
        threshold (int) : max distance between inlier points and dominant plane
        iteration (int) : RANSAC iteration to find dominant plane
    Returns:
        plane (np.ndarray): 1 * 4, (a, b, c, d) that represent the plane ax+by+cz+d = 0
    """
    print("plane.py : Finding the dominat plane...")

    best_inlier = 0
    for ii in range(iteration):
        inlier = 0
        randi = np.random.choice(range(np.shape(X3Dref)[0]), 3, replace=False)
        u = X3Dref[randi[0]]-X3Dref[randi[1]]
        v = X3Dref[randi[0]]-X3Dref[randi[2]]
        normal = np.cross(u, v)
        normal = normal/np.linalg.norm(normal)
        if normal[2] < 0:
            normal = -normal
        d = -np.dot(normal, X3Dref[randi[0]])
        for i in range(len(X3Dref)):
            distance = np.abs(np.dot(normal, X3Dref[i])+d)
            if distance < threshold:
                inlier += 1
        if inlier > best_inlier:
            print("plane.py : dominant plane updated",
                  '(iteration', ii, '/', iteration, ')')
            best_inlier = inlier
            plane = np.append(normal, d)  # [a,b,c,d] ; ax+by+cz+d = 0
    print("plane.py : # of inlier points in dominant plane = ", best_inlier)

    return plane


def make3Dgrid(plane, X3Dref, img, K):
    """_summary_
    plot 3D plane with 3D points in world coordinate
    B = (# of mathced points)
    n = (# of grid points manually selected with wi and hi)
    Args:
        plane (np.ndarray): 1 * 4, (a, b, c, d) that represent the plane ax+by+cz+d = 0
        X3Dref (np.ndarray): B * 3, 3D coordinates relative to second camera
        K (np.ndarray): 3 * 3, intrinsic matrix of camera
    Returns:
        grid3D (np.ndarray): h * w * 3, 3D coordinates of vertically grid points in dominant plane
    """
    print("plane.py : 3D Plot the dominat plane...")

    (imgh, imgw) = img.shape[0:2]

    c = np.array([K[0][2], K[1][2]])
    f = np.array([K[0][0], K[1][1]])

    # image (0,0) and (h,w) to normal image
    vertex_norImg = np.array([[-c[0]/f[0], -c[1]/f[1]],
                              [(imgw-c[0])/f[0], (imgh-c[1])/f[1]]])
    # constant multiple to project to plane
    k = -plane[3]/(plane[0]*vertex_norImg[:, 0] + plane[1]
                   * vertex_norImg[:, 1] + plane[2])

    # (k*a,k*b,z) in the plane is the point that projects to image vertex point (0,0) and (h,w)
    vertex3Dxy = (k*vertex_norImg.T).T
    vertex3Dz = -(plane[0]*vertex3Dxy[:, 0]+plane[1]
                  * vertex3Dxy[:, 1]+plane[3])/plane[2]
    vertex3D = np.concatenate((vertex3Dxy, np.expand_dims(vertex3Dz, 1)), 1)

    init3Dxy = np.array([k[0]*vertex_norImg[0, :], k[0]
                        * vertex_norImg[0, :]-[0, 5]])
    init3Dz = -(plane[0]*init3Dxy[:, 0]+plane[1]
                * init3Dxy[:, 1]+plane[3])/plane[2]
    init3D = np.concatenate((init3Dxy, np.expand_dims(init3Dz, 1)), 1)

    u_grid = init3D[0]-init3D[1]
    u_grid = u_grid/np.linalg.norm(u_grid)
    v_grid = np.cross(u_grid, plane[0:3])
    v_grid = v_grid/np.linalg.norm(v_grid)
    grid3D = [init3D[0]]

    diagvec = vertex3D[1, :]-vertex3D[0, :]
    diag_proj_ugrid = np.dot(diagvec, u_grid)*u_grid
    diag_proj_vgrid = np.dot(diagvec, v_grid)*v_grid

    h = int(np.linalg.norm(diag_proj_ugrid)/5)+2
    w = int(np.linalg.norm(diag_proj_vgrid)/5)+2

    for hi in range(h):
        for wi in range(w):
            if wi == 0 and hi == 0:
                continue  # init3D[0] is already added to palne3Dgrid
            gridij = [init3D[0]+u_grid*5*hi+v_grid*5*wi]
            grid3D = np.concatenate((grid3D, gridij))

    grid3D = grid3D[:, np.newaxis]
    grid3D = np.reshape(grid3D, (h, w, 3))

    return grid3D


def plot3Dplane(plane, grid3D, X3Dref):
    """_summary_
    plot 3D plane with 3D points in world coordinate
    B = (# of mathced points)
    n = (# of grid points manually selected with wi and hi)
    Args:
        plane (np.ndarray): 1 * 4, (a, b, c, d) that represent the plane ax+by+cz+d = 0
        X3Dref (np.ndarray): B * 3, 3D coordinates relative to second camera
        K (np.ndarray): 3 * 3, intrinsic matrix of camera
    Returns:
        grid3D (np.ndarray): h * w * 3, 3D coordinates of vertically grid points in dominant plane
    """
    print("plane.py : 3D Plot the dominat plane...")

    h = grid3D.shape[0]
    w = grid3D.shape[1]
    grid3D = np.reshape(grid3D, (h*w, 3))

    meshx_plane, meshy_plane = np.meshgrid(range(-30, 20), range(-20, 20))
    meshz_plane = -(plane[0]*meshx_plane+plane[1]
                    * meshy_plane+plane[3])/plane[2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("XYZ coordinates relative to second camera")
    ax.set_xlabel("x")
    ax.set_xlim(-40, 40)
    ax.set_ylabel("y")
    ax.set_ylim(-40, 20)
    ax.set_zlabel("z")
    ax.set_zlim(0, 100)
    ax.plot_surface(meshx_plane, meshy_plane, meshz_plane, alpha=0.2)
    # ax.scatter(grid3D[:,0],grid3D[:,1],grid3D[:,2],marker='.', s=10)

    for i in range(h):
        ax.plot([grid3D[w*i][0], grid3D[w*(i+1)-1][0]], [grid3D[w*i][1], grid3D[w*(i+1)-1]
                [1]], zs=[grid3D[w*i][2], grid3D[w*(i+1)-1][2]], color='0.5', linewidth=1)
    for j in range(w):
        ax.plot([grid3D[j][0], grid3D[(h-1)*w+j][0]], [grid3D[j][1], grid3D[(h-1)*w+j]
                [1]], zs=[grid3D[j][2], grid3D[(h-1)*w+j][2]], color='0.5', linewidth=1)
    ax.scatter(X3Dref[:, 0], X3Dref[:, 1], X3Dref[:, 2], marker='o', s=15)
    plt.show()


def obj3Dto2D(X3Dref, K):
    c = np.array([K[0][2], K[1][2]])
    f = np.array([K[0][0], K[1][1]])

    X3Dref = np.matrix(X3Dref)
    objImg = X3Dref/X3Dref[:, 2]
    obj_norImgxy = np.array(objImg[:, 0:2])
    objImg = obj_norImgxy*f+c

    return objImg


def plot2Dplane(keyImg, K, X3D, grid3D):
    """_summary_
    plot 3D plane with 3D points in world coordinate
    B = (# of mathced points)
    n = (# of grid points manually selected with wi and hi)
    Args:
        keyImg: image that represent the world coordinate
        K (np.ndarray): 3 * 3, intrinsic matrix of camera
        X3Dref (np.ndarray): B * 3, 3D coordinates relative to second camera
        X3Dref (np.ndarray): h * w * 3, 3D coordinates of vertically grid points in dominant plane
    Returns:
        None
    """
    print("plane.py : Plot the dominant plane grid in 2D image ...")

    h = grid3D.shape[0]
    w = grid3D.shape[1]
    grid3D = np.reshape(grid3D, (h*w, 3))

    objImg = obj3Dto2D(grid3D, K)

    plt.imshow(keyImg)
    # grid vertex plot
    # plt.scatter(objImg[:,0], objImg[:,1],s=10)

    for i in range(h):
        plt.plot([objImg[w*i][0], objImg[w*(i+1)-1][0]], [objImg[w*i]
                 [1], objImg[w*(i+1)-1][1]], color='r', linewidth=1)
    for j in range(w):
        plt.plot([objImg[j][0], objImg[(h-1)*w+j][0]], [objImg[j]
                 [1], objImg[(h-1)*w+j][1]], color='r', linewidth=1)

    # plot feature
    X2D0 = obj3Dto2D(X3D, K)

    plt.scatter(X2D0[:, 0], X2D0[:, 1], s=10)
    plt.axis("off")
    plt.title("Select some points where you want to display cubes")
    X2Dref = []

    print("plane.py : click the reference points to place the cubes...")
    def onclick(event):
        if event.button == 1:  # Check if left mouse button is X2Dref
            x = int(event.xdata)
            y = int(event.ydata)
            X2Dref.append([x, y])
            print(f"plane.py : clicked pixel coordinates: ({x}, {y})")

    cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    return X2Dref


def makeCube(X2Dref, plane, K):

    c = np.array([K[0][2], K[1][2]])
    f = np.array([K[0][0], K[1][1]])

    X2Dref = np.array(X2Dref)

    X2Dnorm = (X2Dref-c)/f

    # constant multiple to project to plane
    k = -plane[3]/(plane[0]*X2Dnorm[:, 0]+plane[1]*X2Dnorm[:, 1]+plane[2])
    k = np.expand_dims(k, axis=0)
    # (-k*a,-k*b,z) in the plane is the point that projects to image point (0,0)
    X3Dxy = k.T*X2Dnorm
    X3Dz = -(plane[0]*X3Dxy[:, 0]+plane[1]*X3Dxy[:, 1]+plane[3])/plane[2]
    X3Dref = np.concatenate((X3Dxy, np.expand_dims(X3Dz, 1)), 1)

    init2D = np.array([-c[0]/f[0], -c[1]/f[1]])

    k = -plane[3]/(plane[0]*init2D[0] + plane[1]
                   * init2D[1] + plane[2])

    init3Dxy = np.array([k*init2D, k*init2D-[0, 5]])
    init3Dz = -(plane[0]*init3Dxy[:, 0]+plane[1]
                * init3Dxy[:, 1]+plane[3])/plane[2]
    init3D = np.concatenate((init3Dxy, np.expand_dims(init3Dz, 1)), 1)

    u_grid = init3D[0]-init3D[1]
    u_grid = u_grid/np.linalg.norm(u_grid)
    v_grid = np.cross(u_grid, plane[0:3])
    v_grid = v_grid/np.linalg.norm(v_grid)

    length = 5
    cube3D = []
    for i in range(X3Dref.shape[0]):
        cube3D_down = [X3Dref[i]-u_grid*length/2-v_grid*length/2,
                       X3Dref[i]-u_grid*length/2+v_grid*length/2,
                       X3Dref[i]+u_grid*length/2+v_grid*length/2,
                       X3Dref[i]+u_grid*length/2-v_grid*length/2]
        cube3D_up = cube3D_down-plane[0:3]*length
        cube3Dall = np.concatenate((cube3D_down, cube3D_up), 0)
        cube3D.append(cube3Dall)

    return cube3D


def get_plane_cube(M, F0, K):
    M.normal_vector = planeRANSAC(M.X_3D_0, 100, 1)
    grid3D = make3Dgrid(M.normal_vector, M.X_3D_0, F0, K)
    plot3Dplane(M.normal_vector, grid3D, M.X_3D_0)
    X2Dref = plot2Dplane(F0, K, M.X_3D_0, grid3D)
    cube3D = makeCube(X2Dref, M.normal_vector, K)
    M.X_3D_ref = cube3D
    return M
