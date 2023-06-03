import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def plot_cube(M, C):

    vertices = M.X_3D_ref

    plt.scatter(vertices[:, 0], vertices[:, 1], 'ro', s=10)

    for j in range(3):
        for i in range(4):
            if j < 2:
                if i < 3:
                    plt.plot([vertices[i+4*j][0], vertices[i+4*j+1][0]], [vertices[i+4*j]
                             [1], vertices[i+4*j+1][1]], color='r', linewidth=2)
                else:
                    plt.plot([vertices[i+4*j][0], vertices[i+4*(j-1)+1][0]], [vertices[i+4*j]
                             [1], vertices[i+4*(j-1)+1][1]], color='r', linewidth=2)
            elif j == 2:
                plt.plot([vertices[i][0], vertices[i+4][0]], [vertices[i]
                         [1], vertices[i+4][1]], color='r', linewidth=2)

    plt.show()

    return
