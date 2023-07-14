# In the name of Allah
import cv2
import numpy as np
from collections import Counter

img = cv2.imread('files/00tennisballs1-superJumbo.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (7, 7), 0)

edges = cv2.Canny(blur, 15, 30)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

light_red = np.array([100, 100, 70])
dark_red = np.array([255, 255, 255])

maskr = cv2.inRange(hsv, light_red, dark_red)

circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1.5, 130, maxRadius=130)

if circles is not None:
	circles = circles[0].astype(np.uint32)

	print(f'There are {len(circles)} balls there')

	for circle in circles:
		cv2.circle(img, (circle[0], circle[1]), circle[2], (0, 0, 255), 2)
		circle_pixels = maskr[circle[1] - circle[2]:circle[1] + circle[2], circle[0] - circle[2]:circle[0] + circle[2]].ravel()
		counter = Counter(circle_pixels)
		if counter[255] > counter[0]:
			print(f'Coordinates of the center of the red ball is: {circle[0]}, {circle[1]}')

cv2.destroyAllWindows()