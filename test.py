import cv2
import numpy as np

# Load the image
img = cv2.imread('road.jpg')

# Define source points
src_pts = np.float32([[0, 0], [img.shape[1]-1, 0], [0, img.shape[0]-1], [img.shape[1]-1, img.shape[0]-1]])

# Define destination points
dst_pts = np.float32([[50, 50], [img.shape[1]-51, 50], [50, img.shape[0]-51], [img.shape[1]-51, img.shape[0]-51]])

# Compute the perspective transform matrix
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# Apply the perspective transformation
img_transformed = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

# Display the images
cv2.imshow('Original', img)
cv2.imshow('Transformed', img_transformed)

cv2.waitKey(0)
cv2.destroyAllWindows()