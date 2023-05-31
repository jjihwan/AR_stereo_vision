import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def obj3Dto2D(obj3D, K):
    c = np.array([K[0][2],K[1][2]])
    f = np.array([K[0][0],K[1][1]])

    obj3D = np.matrix(obj3D)
    objImg = obj3D/obj3D[:,2]
    obj_norImgxy = np.array(objImg[:,0:2])
    objImg = obj_norImgxy*f+c

    return objImg

def plot2D(keyImg, K, X3D, obj3D, category):
    """_summary_
    plot 3D plane with 3D points in world coordinate
    B = (# of mathced points)
    n = (# of grid points manually selected with wi and hi)
    Args:
        keyImg: image that represent the world coordinate
        K (np.ndarray): 3 * 3, intrinsic matrix of camera
        X3D (np.ndarray): B * 3, 3D coordinates relative to second camera
        obj3D (np.ndarray): h * w * 3, 3D coordinates of vertically grid points in dominant plane
    Returns:
        None
    """
    print("plane.py : Plot the dominant plane grid in 2D image ...")
    
    if category == 'grid':
        h = obj3D.shape[0]
        w = obj3D.shape[1]
        obj3D = np.reshape(obj3D, (h*w,3))

        objImg = obj3Dto2D(obj3D, K)
        
        plt.imshow(keyImg)
        # grid vertex plot
        # plt.scatter(objImg[:,0], objImg[:,1],s=10)
        for i in range(h):
            plt.plot([objImg[w*i][0],objImg[w*(i+1)-1][0]],[objImg[w*i][1],objImg[w*(i+1)-1][1]],color='0.5', linewidth=1)
        for j in range(w):
            plt.plot([objImg[j][0],objImg[(h-1)*w+j][0]],[objImg[j][1],objImg[(h-1)*w+j][1]],color='0.5', linewidth=1)
    if category == 'cube':
        objImg = obj3Dto2D(obj3D, K)
        plt.imshow(keyImg)

        plt.scatter(objImg[:,0], objImg[:,1],s=10)
        for j in range(3):
            for i in range(4):        
                if j < 2:
                    if i < 3:
                        plt.plot([objImg[i+4*j][0],objImg[i+4*j+1][0]],[objImg[i+4*j][1],objImg[i+4*j+1][1]],color='0.5',linewidth=1)
                    else:
                        plt.plot([objImg[i+4*j][0],objImg[i+4*(j-1)+1][0]],[objImg[i+4*j][1],objImg[i+4*(j-1)+1][1]],color='0.5',linewidth=1)
                elif j == 2:
                    plt.plot([objImg[i][0],objImg[i+4][0]],[objImg[i][1],objImg[i+4][1]],color='0.5',linewidth=1)

    # plot feature
    X3D = np.matrix(X3D)
    X3D = X3D/X3D[:,2]
    XnorImg = np.array(X3D[:,0:2])
    XImg = XnorImg*f+c

    plt.scatter(XImg[:,0],XImg[:,1],s=10)
    plt.axis("off")
    plt.show()