import numpy as np
import cv2 as cv

def make_video(images):
    print('video.py : making video file...')
    output_video_path = 'output_video.mp4'
    height = images[0].shape[0]
    width = images[0].shape[1]
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # You can choose the desired codec (e.g., 'XVID', 'MJPG', 'DIVX')
    output_video = cv.VideoWriter(output_video_path, fourcc, 10.0, (width, height))

    # Iterate over the images in the directory
    for img in images:
        output_video.write(img)

    # Release the video writer
    output_video.release()
    print('video.py : video file generated!')