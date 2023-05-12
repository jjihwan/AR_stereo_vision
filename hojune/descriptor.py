import numpy as np
import cv2

def SURF(gray1, gray2):
    # SURF 서술자 추출기 생성 ---①
    detector = cv2.xfeatures2d.SURF_create()
    kp1, desc1 = detector.detectAndCompute(gray1, None)
    kp2, desc2 = detector.detectAndCompute(gray2, None)

    # BFMatcher 생성, L2 거리, 상호 체크 ---③
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # 매칭 계산 ---④
    matches = matcher.match(desc1, desc2)
    return kp1, kp2, matches

def SIFT(gray1, gray2):
    # SIFT로 서술자 추출 ---①
    detector = cv2.xfeatures2d.SIFT_create()
    kp1, desc1 = detector.detectAndCompute(gray1, None)
    kp2, desc2 = detector.detectAndCompute(gray2, None)
    print('key point extracted')
    # BF-Hamming으로 매칭 ---②
    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = matcher.match(desc1, desc2)

    # 매칭 결과를 거리기준 오름차순으로 정렬 ---③
    matches = sorted(matches, key=lambda x:x.distance)
    # 최소 거리 값과 최대 거리 값 확보 ---④
    min_dist, max_dist = matches[0].distance, matches[-1].distance
    # 최소 거리의 20% 지점을 임계점으로 설정 ---⑤
    ratio = 0.3
    good_thresh = (max_dist - min_dist) * ratio + min_dist

    # 임계점 보다 작은 매칭점만 좋은 매칭점으로 분류 ---⑥
    good_matches = [m for m in matches if m.distance < good_thresh]
    print('matches:%d/%d, min:%.2f, max:%.2f, thresh:%.2f' \
            %(len(good_matches),len(matches), min_dist, max_dist, good_thresh))
    return kp1, kp2, good_matches

def ORB(gray1, gray2):
    # ORB로 서술자 추출 ---①
    detector = cv2.ORB_create()
    kp1, desc1 = detector.detectAndCompute(gray1, None)
    kp2, desc2 = detector.detectAndCompute(gray2, None)
    # BF-Hamming으로 매칭 ---②
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc1, desc2)

    # 매칭 결과를 거리기준 오름차순으로 정렬 ---③
    matches = sorted(matches, key=lambda x:x.distance)
    # 최소 거리 값과 최대 거리 값 확보 ---④
    min_dist, max_dist = matches[0].distance, matches[-1].distance
    # 최소 거리의 20% 지점을 임계점으로 설정 ---⑤
    ratio = 0.3
    good_thresh = (max_dist - min_dist) * ratio + min_dist

    # 임계점 보다 작은 매칭점만 좋은 매칭점으로 분류 ---⑥
    good_matches = [m for m in matches if m.distance < good_thresh]
    print('matches:%d/%d, min:%.2f, max:%.2f, thresh:%.2f' \
            %(len(good_matches),len(matches), min_dist, max_dist, good_thresh))
    return kp1, kp2, good_matches
