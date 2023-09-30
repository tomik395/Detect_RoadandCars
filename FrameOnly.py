import cv2
import numpy as np

#image filtering parameters
guassian_window = 5
max_slope = 0.50
edge_low = 50
edge_high = 150
mask_height = 1.5

#line detection parameters
line_detection_rho = 15
line_detection_theta = np.pi/180
line_detection_threshold = 200
line_detection_minLine = 10
line_detection_maxGap = 5

#car detection parameters
car_minsize = 80

#   IMAGE FILTERING ----------------
roadImage_original = cv2.imread('road.jpg') #read image
roadImage_grayscale = cv2.cvtColor(roadImage_original, cv2.COLOR_BGR2GRAY) #transform to GC
roadImage_blur = cv2.GaussianBlur(roadImage_grayscale, (guassian_window, guassian_window), 0) #5-pixel window blur to GC_blur
roadImage_edges = cv2.Canny(roadImage_blur, edge_low, edge_high) #use Canny to get edges

height, width = roadImage_original.shape[:2]

offset = width / 12
top_left = (width / 2 - offset, height / mask_height)
top_right = (width / 2 + offset, height / mask_height)
mid_left = (width / 8, height - width / 8)
mid_right = (width - width / 8, height - width / 8)
bottom_left = (width / 8, height)
bottom_right = (width - width / 8, height)
vertices = np.array([[bottom_left, mid_left, top_left, top_right, mid_right, bottom_right]], dtype=np.int32)

mask = np.zeros_like(roadImage_edges)
cv2.fillPoly(mask, vertices, 255)
roadImage_edges_masked = cv2.bitwise_and(roadImage_edges, mask)  # get masked version of image


cv2.imshow('roadImage original', roadImage_original)
cv2.waitKey(0)
cv2.imshow('roadImage edges', roadImage_edges)
cv2.imwrite('edges_image.jpg', roadImage_edges)
cv2.waitKey(0)
cv2.imshow('roadImage edges masked', roadImage_edges_masked)
cv2.imwrite('edges_image_masked.jpg', roadImage_edges_masked)
cv2.waitKey(0)

#   LINE DETECTION ----------------
lines = cv2.HoughLinesP(roadImage_edges_masked, line_detection_rho, line_detection_theta, line_detection_threshold, None, line_detection_minLine, line_detection_maxGap) #detect lines
print("AMOUNT OF LINES DETECTED: " + str(len(lines)))
for line in lines:
    for x1,y1,x2,y2 in line:
        if abs((y2-y1)/(x2-x1)) > max_slope: # remove line if slope is near horizontal
            cv2.line(roadImage_original, (x1,y1), (x2,y2), (20, 220, 20), 3) #draw line on original image


#   CAR DETECTION ----------------
car_cascade = cv2.CascadeClassifier('cars.xml') #import 'car' data
#find things that look like cars
cars = car_cascade.detectMultiScale(roadImage_original, scaleFactor=1.1, minNeighbors=5, minSize=(car_minsize, car_minsize))
print("AMOUNT OF CARS DETECTED: " + str(len(cars)))
for x,y,width,height in cars:
    cv2.rectangle(roadImage_original, (x+10, y+10), (x+width-10, y+height-10), (0, 0, 255), 2) #draw rectangle around 'car'

#   RESULTS ----------------
cv2.imwrite('resulting_image.jpg', roadImage_original)
cv2.imshow('Detected Image Overlay', roadImage_original)
cv2.waitKey(0)
cv2.destroyAllWindows()