import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import argparse
from core.tracking import get_camera_coordinate, get_normal_coordinate, get_img_coordinate


class FeaturePoints:
    def __init__(self, X_3D_0):
        self.X_3D_0 = X_3D_0
        self.X_2D_prev = None
        self.X_3D_prev = None
        self.X_2D_cur = None


def optical_flow(Fn1, Fn2, X_3D_0, C1):

    FP = FeaturePoints(X_3D_0)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(50, 50),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    Fn1_gray = cv.cvtColor(Fn1, cv.COLOR_RGB2GRAY)
    Fn2_gray = cv.cvtColor(Fn2, cv.COLOR_RGB2GRAY)

    X3D0 = FP.X_3D_0
    pose = C1.pose

    X3D1 = get_camera_coordinate(pose, X3D0)
    X2Dn1 = get_normal_coordinate(X3D1)
    X2D1 = get_img_coordinate(X2Dn1, C1.K)

    X2D1 = np.float32(np.expand_dims(X2D1, 1))

    # (n, 1, 2)
    X2D2, status, err = cv.calcOpticalFlowPyrLK(
        Fn1_gray, Fn2_gray, X2D1, None, **lk_params)

    estimated_X3D2 = get_camera_coordinate(C1.motion, X3D1)
    estimated_X2Dn2 = get_normal_coordinate(estimated_X3D2)
    estimated_X2D2 = get_img_coordinate(estimated_X2Dn2, C1.K)

    k = 2.5
    error = np.linalg.norm(estimated_X2D2 - X2D2.squeeze(), axis=1)
    error = error[:, None]
    avg = error.mean()
    std = error.std()
    is_inlier = np.equal(((avg - k * std) <= error),
                         (error <= (avg + k * std)))
    # print(is_inlier.shape)
    is_inlier = np.logical_or(is_inlier, error <= 5)

    status = is_inlier & status
    # print(np.sum(status))
    # (n, 1, 3)
    X3D1 = np.expand_dims(X3D1, 1)
    X3D0 = np.expand_dims(X_3D_0, 1)

    if X2D2 is not None:
        X2D2good = X2D2[status == 1]
        X2D1good = X2D1[status == 1]
        X3D1good = X3D1[status == 1]
        X3D0good = X3D0[status == 1]

    FP.X_3D_0 = X3D0good
    FP.X_2D_prev = X2D1good
    FP.X_3D_prev = X3D1good
    FP.X_2D_cur = X2D2good

    # Fn1_BGR = cv.cvtColor(Fn1, cv.COLOR_RGB2BGR)
    # color = np.random.randint(0, 255, (200, 3))
    # mask = np.zeros_like(Fn1)
    # for i, (new, old) in enumerate(zip(X2D2good, X2D1good)):
    #     a, b = new.ravel()
    #     c, d = old.ravel()
    #     mask = cv.line(mask, (int(a), int(b)),
    #                    (int(c), int(d)), color[i].tolist(), 2)
    #     frame = cv.circle(Fn1_BGR, (int(a), int(b)), 5, color[i].tolist(), -1)
    # img1 = cv.add(Fn1_BGR, mask)
    # cv.imshow('frame_prev', img1)
    # cv.waitKey(0)

    # Fn2_BGR = cv.cvtColor(Fn2, cv.COLOR_RGB2BGR)
    # color = np.random.randint(0, 255, (5000, 3))
    # mask = np.zeros_like(Fn2)
    # for i, (new, old) in enumerate(zip(X2D2good, X2D1good)):
    #     a, b = new.ravel()
    #     c, d = old.ravel()
    #     mask = cv.line(mask, (int(a), int(b)),
    #                    (int(c), int(d)), color[i].tolist(), 2)
    #     frame = cv.circle(Fn2_BGR, (int(a), int(b)), 5, color[i].tolist(), -1)
    # img = cv.add(Fn2_BGR, mask)
    # cv.imshow('frame_cur', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return FP
