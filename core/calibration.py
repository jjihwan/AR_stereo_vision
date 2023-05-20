import cv2 as cv
import numpy as np
import glob

def calibration(imgPath):
    print('calibration.py : calibrating started')
    chessboard = (6,8) # checker board column row
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # checker 3d point
    objpoints = []
    # checker 2d point
    imgpoints = [] 
    # 3D world coordinate
    objp = np.zeros((1, chessboard[0] * chessboard[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)
    prev_img_shape = None
    images = glob.glob(imgPath)
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        # find checker board corner
        # if the # of corner is satisfied, ret = true
        ret, corners = cv.findChessboardCorners(gray,
                                                chessboard,
                                                cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
        print('calibration.py : chessboard detect', ret)
        if ret == True:
            objpoints.append(objp)
            # refine pixel coordinate
            corners2 = cv.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            imgpoints.append(corners2)
            img = cv.drawChessboardCorners(img, chessboard, corners2, ret)
        # if you do not want to check image, erase under 3 lines
    #     cv.imshow('img',img)
    #     cv.waitKey(0)
    # cv.destroyAllWindows()
    # ret, K, distortion, rotation, translation
    ret, K, dist, R, t = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    h, w = img.shape[:2] # 480, 640
    newK, roi = cv.getOptimalNewCameraMatrix(K, dist, (w,h), 0, (w,h))
    return K, dist, newK, roi

def undistortImg(img, K, dist, newK, roti):
    undistImg = cv.undistort(img, K, dist, None, newK)
    # crop the image
    x, y, w, h = roi
    undistImg = undistImg[y:y+h, x:x+w]
    return undistImg

# K, dist, newK, roi = calibration('./data/calibration/*.jpeg')