import argparse
import os
import numpy as np
import cv2 as cv
<<<<<<< HEAD
from core.map_initialization import map_init
from core.plane import planeRANSAC, plot_plane
from core.calibration import calibration
from core.cube import makeCube
from core.plot import plot3Dto2D

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--img1", "-i1",  type=str, default="./core/data/all1.jpeg", help="Directory of first image for map initialization")
    parser.add_argument("--img2", "-i2",  type=str, default="./core/data/all2.jpeg", help="Directory of second image for map initialization") 
    parser.add_argument("--NNDR_RATIO", "-nndr",  type=float, default=0.7, help="Threshold for Nearest Neighbor Distance Ratio")
    args = parser.parse_args()
    
    ### calibration
    # If you need to calibrate,
    # K, dist, newK, roi = calibration('./core/data/calibration/*.jpeg', 6, 8) # (image path, gridx, gridy)
    # print('K =',K)

    # Or if you already calibrated, just put K
    K = [[3.10593801e+03, 0.00000000e+00, 1.53552466e+03],
    [0.00000000e+00, 3.08841292e+03, 2.03002207e+03],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]


    ### Initialize a Map, which is a set of 3D coordinates of feature points 
    img1, X1, X2, X3D = map_init(args.img1, args.img2, args.NNDR_RATIO, K)

    dom_plane = planeRANSAC(X3D, 100, 0.05) # X3D, iteration, threshold

    # Plot the plane,
    planeGrid3D = plot_plane(dom_plane, X3D, K)
    plot3Dto2D(img1, K, X3D, planeGrid3D, 'grid')

    # Click point & Plot cube
    # obj3D = makeCube(img1, dom_plane, K, X3D)
    # plot3Dto2D(img1, K, X3D, obj3D, 'cube')

    ### TBC ...
    
=======
from core import worker


def get_video(path_to_video):
    print("main.py : Get video ...")
    V = cv.VideoCapture(path_to_video)

    frames = []
    ret = True
    i = 0

    while ret:
        ret, img = V.read()  # read one frame from the 'capture' object; img is (H, W, C)
        if ret:
            if i == 0 or (i % 15 == 0 and i >= 150):
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                frames.append(img)
            i = i + 1
    video = np.stack(frames, axis=0)  # dimensions (T, H, W, C)
    print("main.py : Total", video.shape[0], "frames detected")
    return video


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", "-v", type=str, default="./core/data/short.MOV",
                        help="Directory for video")
    parser.add_argument("--img1", "-i1",  type=str, default="./core/data/desk1.jpeg",
                        help="Directory of first image for map initialization")
    parser.add_argument("--img2", "-i2",  type=str, default="./core/data/desk2.jpeg",
                        help="Directory of second image for map initialization")
    parser.add_argument("--NNDR_RATIO", "-nndr",  type=float, default=0.7,
                        help="Threshold for Nearest Neighbor Distance Ratio")
    parser.add_argument("--calibration", "-c", type=bool, default=False,
                        const=True, nargs='?', help="Do you want to calibrate your new camera?")
    args = parser.parse_args()

    video = get_video(args.video)

    worker.work(video, args)
>>>>>>> d99a1324a6e93970d1932cae412a49c514f72f36
