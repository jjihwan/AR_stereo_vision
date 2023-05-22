import argparse
import os
import numpy as np
from core.map_initialization import map_init
from core.plane import planeRANSAC, plot_plane, plane3Dto2D
from core.calibration import calibration

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--img1", "-i1",  type=str, default="./core/data/desk1.jpeg", help="Directory of first image for map initialization")
    parser.add_argument("--img2", "-i2",  type=str, default="./core/data/desk2.jpeg", help="Directory of second image for map initialization") 
    parser.add_argument("--NNDR_RATIO", "-nndr",  type=float, default=0.7, help="Threshold for Nearest Neighbor Distance Ratio")
    args = parser.parse_args()
    
    ### calibration
    # If you need to calibrate,
    # K, dist, newK, roi = calibration('./core/data/calibration/*.jpeg', 6, 8) # (image path, gridx, gridy)
    # print('K =',K)

    # Or if you already calibrated, just put K
    K = np.array([[3.10593801e+03, 0.00000000e+00, 1.53552466e+03],
    [0.00000000e+00, 3.08841292e+03, 2.03002207e+03],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    ### Initialize a Map, which is a set of 3D coordinates of feature points 
    img2, X1, X2, X3D = map_init(args.img1, args.img2, args.NNDR_RATIO, K)

    dom_plane = planeRANSAC(X3D, 100, 0.05) # X3D, iteration, threshold
    planeGrid3D = plot_plane(dom_plane, X3D, K)
    plane3Dto2D(img2, K, X3D, planeGrid3D)

    ### TBC ...
    