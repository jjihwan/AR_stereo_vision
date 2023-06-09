import argparse
import os
import numpy as np
import cv2 as cv
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
            if i == 0 or (i % 1 == 0 and i >= 120):
                # cv.imwrite(str(i)+'.jpeg',img)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                frames.append(img)
            i = i + 1
    video = np.stack(frames, axis=0)  # dimensions (T, H, W, C)
    print("main.py : Total", video.shape[0], "frames detected")
    return video


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", "-v", type=str, default="./core/data/short.MOV",
                        help="Directory for input video")
    parser.add_argument("--img1", "-i1",  type=str, default="./core/data/desk1.jpeg",
                        help="Directory of first image for map initialization")
    parser.add_argument("--img2", "-i2",  type=str, default="./core/data/desk2.jpeg",
                        help="Directory of second image for map initialization")
    parser.add_argument("--output_path", "-o",  type=str, default="./core/results/output_video.mp4",
                        help="Directory for output video")
    parser.add_argument("--NNDR_RATIO", "-nndr",  type=float, default=0.7,
                        help="Threshold for Nearest Neighbor Distance Ratio")
    parser.add_argument("--calibration", "-c", type=bool, default=False,
                        const=True, nargs='?', help="Do you want to calibrate your new camera?")
    parser.add_argument("--dev", "-d", type=bool, default=False,
                        const=True, nargs='?', help="Are you in develop mode?")
    args = parser.parse_args()

    video = get_video(args.video)

    worker.work(video, args)
