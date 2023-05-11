import cv2 as cv
import numpy as np
import os


def load_images(path1="./data/img1.png", path2="./data/img2.png") :
    img1 = cv.imread(path1)
    img2 = cv.imread(path2)
    
    print(img1.shape)
    return


load_images()