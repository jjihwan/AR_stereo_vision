import numpy as np
import cv2 as cv


def make_video(images, path="./core/results/output_video.mp4"):
    print('video.py : making video file...')
    output_video_path = path
    height = images[0].shape[0]
    width = images[0].shape[1]
    # You can choose the desired codec (e.g., 'XVID', 'MJPG', 'DIVX')
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    output_video = cv.VideoWriter(
        output_video_path, fourcc, 60.0, (width, height))

    # Iterate over the images in the directory
    for img in images:
        output_video.write(img)

    # Release the video writer
    output_video.release()
    print('video.py : video file generated!')
