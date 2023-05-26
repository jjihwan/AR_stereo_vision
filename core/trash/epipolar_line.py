import numpy as np
import cv2

def SE3(X1,X2,K):
    K_inv = np.linalg.inv(K)
    normalX1 = np.dot(K_inv, np.concatenate((X1, np.ones(len(X1))), axis=1).T)
    normalX2 = np.dot(K_inv, np.concatenate((X2, np.ones(len(X2))), axis=1).T)
    E, _ = cv2.findEssentialMat(normalX1, normalX2, K)
    _, R, t, _ = cv2.recoverPose(E, X1, X2, K)
    # F = np.dot(np.dot(K_inv.T, E), K_inv)
    
    return E, R, t

def searchRmatch(X1, E):
    ##

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