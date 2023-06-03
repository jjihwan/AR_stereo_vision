import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from core.tracking import get_camera_coordinate


def plot_cube(img, M, C):

    vertices_3D_ref = M.X_3D_ref

    vertices_3D = get_camera_coordinate(C.pose, vertices_3D_ref)
    vertices_2D = vertices_3D[:, :2] / vertices_3D[:, [2]]

    vertices = vertices_2D @ np.array([[C.K[0, 0], 0],
                                      [0, C.K[1, 1]]]) + np.array([C.K[0, 2], C.K[1, 2]])
    # vertices_3D_ref = np.concatenate(
    #     (vertices_3D_ref, np.ones([vertices_3D_ref.shape[0], 1])), 1)
    # vertices_3D = vertices_3D_ref @ C.pose.T
    # vertices_3D = vertices_3D[:, :3]/vertices_3D[:, [3]]
    # vertices_normal = vertices_3D[:, :2]/vertices_3D[:, [2]]

    # vertices = vertices_normal @ np.array([[C.K[0, 0], 0], [0, C.K[1, 1]]]
    #                                       ) + np.array([C.K[0, 2], C.K[1, 2]])
    # print(vertices.shape)
    # plt.scatter(vertices[:, 0], vertices[:, 1], s=10, color='r')

    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting lines
    ])

    for edge in edges:
        point1 = vertices[edge[0]].astype(int)
        point2 = vertices[edge[1]].astype(int)
        cv.line(img, tuple(point1), tuple(point2), (0, 255, 0), 2)

    cv.imshow("Cube", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return
