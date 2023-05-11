import argparse
import os
from core.map_initialization import map_init

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--img1", "-i1",  type=str, default="./core/data/img1.png", help="Directory of first image for map initialization")
    parser.add_argument("--img2", "-i2",  type=str, default="./core/data/img2.png", help="Directory of second image for map initialization") 
    parser.add_argument("--NNDR_RATIO", "-nndr",  type=float, default=0.7, help="Threshold for Nearest Neighbor Distance Ratio")
    args = parser.parse_args()
    
    ### Initialize a Map, which is a set of 3D coordinates of feature points 
    Map = map_init(args.img1, args.img2, args.NNDR_RATIO)
    
    ### RANSAC ...
    ### TBC ...
    