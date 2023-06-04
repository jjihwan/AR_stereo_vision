import numpy as np
import matplotlib.pyplot as plt
import time
from core.calibration import calibration
from core.map_initialization import map_init_from_frames, map_reconstruction_from_frames
from core.plane import get_plane_cube
from core.projection import plot_cube
from core.optical import optical_flow
from core.tracking import tracking
from core.video import make_video


def work(video, args):
    print("worker.py : Start working!")

    if args.calibration:
        K, _, _, _ = calibration(
            './core/data/calibration/video_mode/*.jpeg', 6, 10)  # (image path, gridx, gridy)
        print(K)
    else:
        # New extrinsic parmameters from 1920*1080 video camera
        K = np.array([[1.69428499e+03, 0.00000000e+00, 9.62922539e+02],
                      [0.00000000e+00, 1.70678063e+03, 5.20552346e+02],
                      [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    C = Camera(K)

    M = map_init_from_frames(video[0], video[1], args.NNDR_RATIO, C.K)

    M = get_plane_cube(M, video[0], C.K)
    # print(cube3D)

    print('worker.py : video frames generating...')
    video_frames = []

    for i in range(1, video.shape[0]-1):
        # start = time.time()
        if i == 1:
            FP = optical_flow(video[i], video[i+1], M.X_3D_0, C, args.dev)
            Nfeat0 = FP.X_3D_0.shape[0]

        else:
            FP = optical_flow(video[i], video[i+1], FP.X_3D_0, C, args.dev)
        Nfeat = FP.X_3D_0.shape[0]
        # print(FP.X_3D_0.shape)

        C = tracking(FP, C)

        if Nfeat / Nfeat0 < 0.5:
            FP = map_reconstruction_from_frames(
                video[i], video[i+1], args.NNDR_RATIO, C, FP)
            Nfeat0 = FP.X_3D_0.shape[0]
        img = plot_cube(video[i+1], M, C, args.dev)
        
        # end = time.time()
        # print(f"{end - start:.5f} sec")

        video_frames.append(img)
    make_video(video_frames)
    print('worker.py : video frames generated!')


class Camera:
    def __init__(self, K):
        self.K = K
        self.R = np.eye(3)
        self.t = np.array([[-10], [0], [0]])
        Rt = np.concatenate((self.R, self.t), 1)
        self.pose = np.concatenate((Rt, [[0, 0, 0, 1]]), 0)
        self.motion = np.eye(4)
