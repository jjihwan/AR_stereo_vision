import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def on_mouse(event, x, y, flags, data):
    if event == cv.EVENT_LBUTTONDOWN:
        data.append([x, y])
        print('The reference point is : (%d, %d)' % (x, y))

        cv.destroyAllWindows()


def clickImg(img):
    data = []
    cv.namedWindow('image')

    # if event occurs, on_mouse function runs
    cv.imshow('image', img)
    cv.setMouseCallback('image', on_mouse, data)

    print(data)
    cv.waitKey()

    clicked2D = np.array(data, dtype=float)
    return clicked2D


def makeCube(img, plane, K, X3D):
    print("plot.py : object select...")

    obj2Dimg = np.array(clickImg(img))
    c = np.array([K[0][2], K[1][2]])
    f = np.array([K[0][0], K[1][1]])

    obj2Dnorm = (obj2Dimg-c)/f
    # constant multiple to project to plane
    k = -plane[3]/(plane[0]*obj2Dnorm[:, 0]+plane[1]*obj2Dnorm[:, 1]+plane[2])
    k = np.expand_dims(k, axis=0)
    # (-k*a,-k*b,z) in the plane is the point that projects to image point (0,0)
    obj3Dxy = k.T*obj2Dnorm
    obj3Dz = -(plane[0]*obj3Dxy[:, 0]+plane[1]*obj3Dxy[:, 1]+plane[3])/plane[2]
    obj3D = np.concatenate((obj3Dxy, np.expand_dims(obj3Dz, 1)), 1)
    print(obj3D)
    obj3D = np.concatenate((obj3D, obj3D-plane[0:3]*5), 0)

    meshx_plane, meshy_plane = np.meshgrid(range(-30, 20), range(-20, 20))
    meshz_plane = -(plane[0]*meshx_plane+plane[1]
                    * meshy_plane+plane[3])/plane[2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("XYZ coordinates relative to second camera")
    ax.set_xlabel("x")
    ax.set_xlim(-20, 20)
    ax.set_ylabel("y")
    ax.set_ylim(-20, 20)
    ax.set_zlabel("z")
    ax.set_zlim(0, 50)
    ax.plot_surface(meshx_plane, meshy_plane, meshz_plane, alpha=0.2)
    ax.scatter(obj3D[:, 0], obj3D[:, 1], obj3D[:, 2], marker='.', s=10)
    ax.scatter(X3D[:, 0], X3D[:, 1], X3D[:, 2], marker='o', s=15)
    plt.show()

    return obj3D
