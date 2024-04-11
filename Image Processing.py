#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np

# Function to detect shapes
def detect_shapes(img):
    # Perform shape detection within the ROI
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 150)
    
    # Apply thresholding to isolate white regions
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            # Check if it is a square or rectangle based on aspect ratio
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h

            if 0.9 <= aspect_ratio <= 1.1:
                shape = "Square"
            else:
                shape = "Trapezium"
        elif len(approx) == 5:
            shape = "Pentagon"
        elif len(approx) == 6:
            shape = "Hexagon"
        else:
            shape = "Circle"
        
        if area > 100:
            cv2.drawContours(img, [approx], 0, (0, 0, 0), 2)
            cv2.putText(img, shape, (approx.ravel()[0], approx.ravel()[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return img

# Function to detect objects and their colors
def detect_objects(img, color_ranges, object_counts, object_detected):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    for color, (lower, upper, color_value) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(img, (x, y), (x + w, y + h), color_value, 2)
                if not object_detected[color]:
                    object_counts[color] += 1
                    object_detected[color] = True
                    cv2.putText(img, f"{color.capitalize()} Object {object_counts[color]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_value, 2)
                else:
                    i= object_counts[color]
                    cv2.putText(img, f"{color.capitalize()} Object {i + 1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_value, 2)
   
    return img

# # Create a VideoCapture object for the default camera
cap = cv2.VideoCapture(1)
ret, frame = cap.read()
frame = cv2.resize(frame, (640, 480))
roi = cv2.selectROI(frame)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Failed to open the camera")
    exit()
# # Load the image
# image = cv2.imread("C:/Users/IdeasClinicCoops/Downloads/image.jpeg")

# # Select the ROI
# roi = cv2.selectROI("Select ROI", image)
# Define color ranges for object detection
color_ranges = {
    'red': ([0, 50, 50], [10, 255, 255], (0, 0, 255)),
    'blue': ([100, 50, 50], [130, 255, 255], (255, 0, 0)),
    'green': ([40, 50, 50], [70, 255, 255], (0, 255, 0)),
    'purple': ([130, 50, 50], [160, 255, 255], (128, 0, 128))
}

# # Initialize object count arrays
# object_counts = {
#     'red': 0,
#     'blue': 0,
#     'green': 0,
#     'purple': 0
# }

# Initialize object count and detection flags
object_counts = {color: 0 for color in color_ranges}
object_detected = {color: False for color in color_ranges}

# Start reading frames from the camera
while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    frame = cv2.resize(frame, (640, 480))
    
    # Check if the frame was successfully read
    if not ret:
        print("Failed to read frame from the camera")
#         break
    
    roi_frame = frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]

# # Extract the ROI from the image
# roi_frame = image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]

    # Detect shapes and objects
    shapes_img = detect_shapes(roi_frame.copy())
    objects_img = detect_objects(roi_frame.copy(), color_ranges, object_counts, object_detected)
    
    # Display the frames
    cv2.imshow("Color and Shape Detection", np.hstack((shapes_img, objects_img)))
    
    
    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
           break

# Release the camera and close any open windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:




