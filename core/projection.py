import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from core.tracking import get_camera_coordinate


def plot_cube(img, M, C):
    for vertices_3D_ref in M.X_3D_ref:

        vertices_3D = get_camera_coordinate(C.pose, vertices_3D_ref)
        vertices_2D = vertices_3D[:, :2] / vertices_3D[:, [2]]

        vertices = vertices_2D @ np.array([[C.K[0, 0], 0],
                                           [0, C.K[1, 1]]]) + np.array([C.K[0, 2], C.K[1, 2]])

        edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting lines
        ])

        for edge in edges:
            point1 = vertices[edge[0]].astype(int)
            point2 = vertices[edge[1]].astype(int)
            cv.line(img, tuple(point1), tuple(point2), (0, 255, 0), 2)

    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    cv.imshow("Cube", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return
