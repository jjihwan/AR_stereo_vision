import numpy as np
import cv2

img1 = cv2.imread('../data/bear2.jpeg')
img2 = cv2.imread('../data/bear3.jpeg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ORB로 서술자 추출 ---①
detector = cv2.ORB_create()
kp1, desc1 = detector.detectAndCompute(gray1, None)
kp2, desc2 = detector.detectAndCompute(gray2, None)
# BF-Hamming 생성 ---②
matcher = cv2.BFMatcher(cv2.NORM_HAMMING2)
# knnMatch, k=2 ---③
matches = matcher.knnMatch(desc1, desc2, 2)

# 첫번재 이웃의 거리가 두 번째 이웃 거리의 75% 이내인 것만 추출---⑤
ratio = 0.75
good_matches = [first for first,second in matches \
                    if first.distance < second.distance * ratio]
print('matches:%d/%d' %(len(good_matches),len(matches)))

match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
for match in good_matches:
    # Draw a line for each match
    pt1 = kp1[match.queryIdx].pt
    pt2 = kp2[match.trainIdx].pt
    pt1 = (int(pt1[0]), int(pt1[1]))
    pt2 = (int(pt2[0]+np.shape(img1)[1]), int(pt2[1]))
    match_img = cv2.line(match_img, pt1, pt2, (0, 255, 0), thickness=3)

# Display image
cv2.imshow('Matches', match_img)
cv2.waitKey(0)
cv2.destroyAllWindows()