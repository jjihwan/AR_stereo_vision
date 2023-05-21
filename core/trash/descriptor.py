import numpy as np
import cv2

def SURF(gray1, gray2):
    detector = cv2.xfeatures2d.SURF_create()
    kp1, desc1 = detector.detectAndCompute(gray1, None)
    kp2, desc2 = detector.detectAndCompute(gray2, None)

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(desc1, desc2)
    return kp1, kp2, matches

def SIFT(gray1, gray2):
    detector = cv2.xfeatures2d.SIFT_create()
    kp1, desc1 = detector.detectAndCompute(gray1, None)
    kp2, desc2 = detector.detectAndCompute(gray2, None)
    print('key point extracted')
    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = matcher.match(desc1, desc2)

    # 매칭 결과를 거리기준 오름차순으로 정렬
    matches = sorted(matches, key=lambda x:x.distance)
    min_dist, max_dist = matches[0].distance, matches[-1].distance
    # 최소 거리의 30% 지점을 임계점으로 설정
    ratio = 0.3
    good_thresh = (max_dist - min_dist) * ratio + min_dist

    good_matches = [m for m in matches if m.distance < good_thresh]
    print('matches:%d/%d, min:%.2f, max:%.2f, thresh:%.2f' \
            %(len(good_matches),len(matches), min_dist, max_dist, good_thresh))
    return kp1, kp2, good_matches

def ORB(gray1, gray2):
    detector = cv2.ORB_create()
    kp1, desc1 = detector.detectAndCompute(gray1, None)
    kp2, desc2 = detector.detectAndCompute(gray2, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc1, desc2)

    # 매칭 결과를 거리기준 오름차순으로 정렬
    matches = sorted(matches, key=lambda x:x.distance)
    min_dist, max_dist = matches[0].distance, matches[-1].distance
    # 최소 거리의 30% 지점을 임계점으로 설정
    ratio = 0.3
    good_thresh = (max_dist - min_dist) * ratio + min_dist

    good_matches = [m for m in matches if m.distance < good_thresh]
    print('matches:%d/%d, min:%.2f, max:%.2f, thresh:%.2f' \
            %(len(good_matches),len(matches), min_dist, max_dist, good_thresh))
    return kp1, kp2, good_matches
