#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read in image
img = cv2.imread('/Users/pvss2807/Downloads/P024p2mWOD4BK31HCd3T6K.jpg',0)

# Convert image to grayscale
blurred = cv2.GaussianBlur(img, (5, 5), 0)
# Threshold image to create binary mask
mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
#plt.imshow(mask)
# Find contours in the binary mask
contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# x_coords = []
# y_coords = []
# Loop through contours and analyze each region
for i in range(len(contours)):
    # Calculate the area of the contour
    area = cv2.contourArea(contours[i])
    # Approximate the contour to reduce the number of vertices
    epsilon = 0.05 * cv2.arcLength(contours[i], True)
    approx = cv2.approxPolyDP(contours[i], epsilon, True)
    # Analyze the contour for different geometric shapes
    if len(approx) == 3:
        # Triangle
        print("Triangle with area:", area)
    elif len(approx) == 4:
        # Check if the contour is a square or a rectangle
        x, y, w, h = cv2.boundingRect(contours[i])
        aspect_ratio = float(w)/h
        if aspect_ratio >= 0.95 and aspect_ratio <= 1.05:
            # Square
            print("Square with area:", area)
        else:
            # Rectangle
            print("Rectangle with area:", area)
    elif len(approx) >= 8:
        # Circle or ellipse
        (x, y), (MA, ma), angle = cv2.fitEllipse(contours[i])
        print(MA / ma)
        if MA / ma >= 0.95 and MA / ma <= 1.05:
            # Circle
            print("Circle with area:", area)
        else:
            # Ellipse
            print("Ellipse with area:", area)
    elif len(approx) == 6:
        # Check if all angles are close to 120 degrees
#         angles = []
#         for i in range(6):
#             pt1 = approx[i][0]
#             pt2 = approx[(i + 1) % 6][0]
#             pt3 = approx[(i + 2) % 6][0]
#             v1 = np.array([pt1[0] - pt2[0], pt1[1] - pt2[1]])
#             v2 = np.array([pt3[0] - pt2[0], pt3[1] - pt2[1]])
#             angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
#             angles.append(angle)
#         angles = np.array(angles) * 180 / np.pi
#         if np.allclose(angles, 120, atol=10):
            print('Hexagon with area', area)
#     x_coords.append(contours[i][:, 0, 0])
#     y_coords.append(contours[i][:, 0, 1])
    
plt.imshow(img)
# for i in range(len(x_coords)):
#     # Make sure that the x and y arrays have the same shape
#     x = x_coords[i].reshape(-1, 1)
#     y = y_coords[i].reshape(-1, 1)
#     coords = np.hstack((x, y))
#     plt.plot(coords[:,0], coords[:,1], '-o', color='white', linewidth=2, markersize=2, markerfacecolor='white')
# plt.show()


# In[2]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
# Read the image
image = cv2.imread('/Users/pvss2807/Downloads/P024p2mWOD4BK31HCd3T6K.jpg',0)

# Convert the image to grayscale
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Apply a threshold to segment the image into regions of interest
thresholded_image = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]


# Remove small white regions
img_erode = cv2.erode(thresholded_image, np.ones((3,3), np.uint8), iterations=1)

# Fill gaps between patterns
img_dilate = cv2.dilate(img_erode, np.ones((5,5), np.uint8), iterations=1)
plt.imshow(img_dilate)
# Find contours of the regions of interest
contours, hierarchy = cv2.findContours(img_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
x_coords = []
y_coords = []
# Create a list to store the geometry patterns
patterns = []

# Loop through each contour
for i in range(len(contours)):
    # Compute the area of the contour
    area = cv2.contourArea(contours[i])
    
    # Compute the perimeter of the contour
    perimeter = cv2.arcLength(contours[i], True) 
    
    # Compute the circularity of the contour
    if perimeter == 0 or area == 0:
        circularity = 0
    else:
        circularity =  4 * np.pi * area / (perimeter * perimeter)
    
    # Compute the aspect ratio of the bounding rectangle of the contour
    x,y,w,h = cv2.boundingRect(contours[i])
    aspect_ratio = float(w)/float(h)
    
    # Add the pattern to the list
    patterns.append((area, perimeter, circularity, aspect_ratio))

# Print the list of patterns
#print(patterns)

# Classify each pattern based on its geometry
for pattern in patterns:
    area, perimeter, circularity, aspect_ratio = pattern
    
    if 0.75< circularity < 0.85 and aspect_ratio >= 0.95 and aspect_ratio <= 1.1:
            # Square
            print("Square")
            
    # Check for circles
    elif 0.9 < circularity < 1.1 and abs(area - np.pi * (perimeter/2)**2) < 2000:
        print("Circle")
    
    # Check for rectangles
    elif circularity < 0.5 and abs(area - aspect_ratio * perimeter**2/16) < 2000:
        print("Rectangle")
    
    # Check for triangles
    elif 0.6 < circularity < 0.9 and (area/perimeter) < 10:
        print("Triangle")
        
    elif 0.9 < circularity < 0.95 and abs(area - 3/2 * np.sqrt(3) * (perimeter/6)**2) < 1000:
        print("Hexagon")
    # Check for other shapes
    else:
        print("Other shape")
x_coords.append(contours[i][:, 0, 0])
y_coords.append(contours[i][:, 0, 1])

plt.imshow(image)
for i in range(len(x_coords)):
    # Make sure that the x and y arrays have the same shape
    x = x_coords[i].reshape(-1, 1)
    y = y_coords[i].reshape(-1, 1)
    coords = np.hstack((x, y))
    plt.plot(coords[:,0], coords[:,1], '-o', color='white', linewidth=2, markersize=2, markerfacecolor='white')
plt.show()


# In[3]:


import cv2
import matplotlib.pyplot as plt

# read input image
img = cv2.imread('/Users/pvss2807/Downloads/P034mW_OD4_Bk31HC31d3um_NF.jpg')

# convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# apply bilateral filter to reduce noise and preserve edges
filtered = cv2.bilateralFilter(gray, 9, 75, 75)

# apply adaptive threshold to binarize the image
thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# apply morphological operations to remove small objects and fill holes
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel)

# apply Hough transform to detect circular objects
circles = cv2.HoughCircles(morphed, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=70, param2=30, minRadius=20, maxRadius=50)
contours, hierarchy = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

patterns = []
# draw detected circles on original image
# if circles is not None:
#     circles = np.round(circles[0, :]).astype("int")
#     for (x, y, r) in circles:
#         cv2.circle(img, (x, y), r, (0, 255, 0), 2)
for i in range(len(contours)):
    # Compute the area of the contour
    area = cv2.contourArea(contours[i])
    
    # Compute the perimeter of the contour
    perimeter = cv2.arcLength(contours[i], True) 
    
    # Compute the circularity of the contour
    if perimeter == 0 or area == 0:
        circularity = 0
    else:
        circularity =  4 * np.pi * area / (perimeter * perimeter)
    
    # Compute the aspect ratio of the bounding rectangle of the contour
    x,y,w,h = cv2.boundingRect(contours[i])
    aspect_ratio = float(w)/float(h)
    
    # Add the pattern to the list
    patterns.append((area, perimeter, circularity, aspect_ratio))

# Print the list of patterns
#print(patterns)

# Classify each pattern based on its geometry
for pattern in patterns:
    area, perimeter, circularity, aspect_ratio = pattern
    
    if 0.75< circularity < 0.85 and aspect_ratio >= 0.95 and aspect_ratio <= 1.1:
            # Square
            print("Square")
            
    # Check for circles
    elif 0.9 < circularity < 1.1 and abs(area - np.pi * (perimeter/2)**2) < 2000:
        print("Circle")
    
    # Check for rectangles
    elif circularity < 0.5 and abs(area - aspect_ratio * perimeter**2/16) < 2000:
        print("Rectangle")
    
    # Check for triangles
    elif 0.6 < circularity < 0.9 and (area/perimeter) < 10:
        print("Triangle")
        
    elif 0.9 < circularity < 0.95 and abs(area - 3/2 * np.sqrt(3) * (perimeter/6)**2) < 1000:
        print("Hexagon")
    # Check for other shapes
    else:
        print("Other shape")
# display output image
plt.imshow(morphed)


# In[7]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the black circle image
img = cv2.imread('/Users/pvss2807/Downloads/circle.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Otsu's thresholding to obtain a binary image
thresh_val, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Apply morphological opening to remove small noise and fill in the circle
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# Find the contours in the image
contours, hierarchy = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours onto a new image
shape_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(shape_image, contours, -1, (0,255,0), 2)

# Identify the circle using Hough Circle Transform
circles = cv2.HoughCircles(opened, cv2.HOUGH_GRADIENT, dp=1, minDist=200,
                           param1=50, param2=30, minRadius=0, maxRadius=0)

# Draw the circle onto the shape image
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(shape_image, (i[0], i[1]), i[2], (0, 0, 255), 2)

# Display the shape image
plt.imshow(shape_image)
print(len(circles))


# In[ ]:




