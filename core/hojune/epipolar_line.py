import numpy as np
import cv2
import matplotlib.pyplot as plt
import descriptor
import SE3

img1 = cv2.imread('../data/img1.png')
img2 = cv2.imread('../data/img2.png')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

kp1, kp2, good_matches = descriptor.ORB(gray1,gray2)

points1 = []
points2 = []

for match in good_matches:
    pt1 = kp1[match.queryIdx].pt
    pt2 = kp2[match.trainIdx].pt
    pt1 = [int(pt1[0]), int(pt1[1])]
    pt2 = [int(pt2[0]), int(pt2[1])]
    points1.append(pt1)
    points2.append(pt2)

points1 = np.array(points1)
points2 = np.array(points2)

# intrinsic parameters
f = 1
cx = 0
cy = 0
K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=float)
K_inv = np.linalg.inv(K)

# Normalize the corresponding points
normalized_points1 = np.dot(K_inv, np.concatenate((points1, np.ones((np.shape(good_matches)[0], 1))), axis=1).T)
normalized_points2 = np.dot(K_inv, np.concatenate((points2, np.ones((np.shape(good_matches)[0], 1))), axis=1).T)

# Compute the essential matrix using the normalized corresponding points
E, _ = cv2.findEssentialMat(points1, points2, K)

# Recover the possible rotations and translations from the essential matrix
_, R, t, _ = cv2.recoverPose(E, points1, points2, K)

# Compute the fundamental matrix using the essential matrix and camera intrinsics
F = np.dot(np.dot(np.linalg.inv(K).T, E), np.linalg.inv(K))

p3ds = []
for i in range(np.shape(good_matches)[0]):
    p3d = SE3.match2p3d(R,t, points1[i], points2[i])
    p3ds.append(p3d)

# Compute the epipolar lines in the two views
lines1 = cv2.computeCorrespondEpilines(points2.reshape(-1, 1, 2), 2, E).reshape(-1, 3)
lines2 = cv2.computeCorrespondEpilines(points1.reshape(-1, 1, 2), 1, E).reshape(-1, 3)

# Plot the epipolar lines in the two images
for pt1, pt2, line1, line2 in zip(points1, points2, lines1, lines2):
    # Draw the corresponding points
    cv2.circle(img1, tuple(pt1.astype(int)), 10, (0, 255, 0), -1)
    cv2.circle(img2, tuple(pt2.astype(int)), 10, (0, 255, 0), -1)
    
    # Compute the endpoints of the epipolar lines
    x1, y1 = map(int, [0, -line1[2]/line1[1]])
    x2, y2 = map(int, [img1.shape[1], -(line1[2]+line1[0]*img1.shape[1])/line1[1]])
    u1, v1 = map(int, [0, -line2[2]/line2[1]])
    u2, v2 = map(int, [img2.shape[1], -(line2[2]+line2[0]*img2.shape[1])/line2[1]])
    
    # Draw the epipolar lines
    cv2.line(img1, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.line(img2, (u1, v1), (u2, v2), (0, 0, 255), 3)
    
# Show the images
cv2.imshow('Image 1', img1)
cv2.imshow('Image 2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

for p3d in p3ds:
    ax.scatter(p3d[0], p3d[1], p3d[2], c='r', marker='o', s=15, cmap='Greens')

plt.show()