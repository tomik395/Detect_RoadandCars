import cv2
import numpy as np


#image filtering parameters
guassian_window = 5
max_slope = 0.50
edge_low = 50
edge_high = 150
mask_height = 2.0

#line detection parameters
line_detection_rho = 10
line_detection_theta = np.pi/180
line_detection_threshold = 200
line_detection_minLine = 25
line_detection_maxGap = 5

#car detection parameters
car_minsize = 80

# Load the video
video = cv2.VideoCapture('video.mp4')
fps = video.get(cv2.CAP_PROP_FPS)
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height), isColor=True)
video.set(cv2.CAP_PROP_POS_MSEC, 60000)


# Frame index
frame_index = 0
counter = 0
while True:
    print("Frame: ", counter)
    # Read the video frame by frame
    ret, frame = video.read()

    if not ret:
        break

    roadImage_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # transform to GC
    roadImage_blur = cv2.GaussianBlur(roadImage_grayscale, (guassian_window, guassian_window), 0)  # 5-pixel window blur to GC_blur
    roadImage_edges = cv2.Canny(roadImage_blur, edge_low, edge_high)  # use Canny to get edges

    height, width = frame.shape[:2]

    offset = width/12
    top_left = (width / 2 - offset, height / mask_height)
    top_right = (width / 2 + offset, height / mask_height)
    mid_left = (width/8, height - width/8)
    mid_right = (width - width/8, height - width/8)
    bottom_left = (width/8, height)
    bottom_right = (width - width/8, height)
    vertices = np.array([[bottom_left, mid_left, top_left, top_right, mid_right, bottom_right]], dtype=np.int32)

    mask = np.zeros_like(roadImage_edges)
    cv2.fillPoly(mask, vertices, 255)
    roadImage_edges_masked = cv2.bitwise_and(roadImage_edges, mask)  # get masked version of image

    #   LINE DETECTION ----------------
    lines = cv2.HoughLinesP(roadImage_edges_masked, line_detection_rho, line_detection_theta, line_detection_threshold, None, line_detection_minLine, line_detection_maxGap)  # detect lines

    try:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if abs((y2 - y1) / (x2 - x1)) > max_slope:  # remove line if slope is near horizontal
                    cv2.line(frame, (x1, y1), (x2, y2), (20, 220, 20), 3)  # draw line on original image
    except:
        f = None

    #   CAR DETECTION ----------------
    car_cascade = cv2.CascadeClassifier('cars.xml')  # import 'car' data
    # find things that look like cars
    cars = car_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5,
                                        minSize=(car_minsize, car_minsize))
    for x, y, width, height in cars:
        cv2.rectangle(frame, (x + 10, y + 10), (x + width - 10, y + height - 10), (0, 0, 255),
                      2)  # draw rectangle around 'car'

    out.write(frame)
    counter = counter+1

# Release the video object
video.release()
out.release()


# Close all OpenCV windows
cv2.destroyAllWindows()