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
