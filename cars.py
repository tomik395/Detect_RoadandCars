import cv2

# Load the pre-trained Haar Cascade classifier for cars
car_cascade = cv2.CascadeClassifier('cars.xml')

# Read the image
image = cv2.imread('road.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect cars in the image using the Haar Cascade classifier
cars = car_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))

# Draw rectangles around the detected cars
for (x, y, width, height) in cars:
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

# Show the image with the detected cars
cv2.imshow('Detected Cars', image)
cv2.waitKey(0)
cv2.destroyAllWindows()