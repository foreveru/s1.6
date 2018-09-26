# encoding: utf-8
# 黄色检测
import numpy as np
import argparse
import cv2

image = cv2.imread('./red.jpg')
hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_r = np.array([0, 123, 100])
upper_r = np.array([5, 255, 255])

mask = cv2.inRange(hsv_img, lower_r, upper_r)
dilated = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=2)
circles = cv2.HoughCircles(dilated, cv2.HOUGH_GRADIENT, 1, 100, param1=15, param2=7, minRadius=10, maxRadius=20)

# if circles is not None:
#     x, y, radius = circles[0][0]
#     center = (x, y)
#     cv2.circle(image, center, radius, (0, 255, 0), 2)
#res = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)
cv2.imshow("image", np.hstack([image, hsv_img, dilated]))
cv2.waitKey(0)

