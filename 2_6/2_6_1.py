# In the name of Allah
import cv2
import numpy as np

l = 125
lx1 = 220.
ly1 = 170.
lx2 = 400.
ly2 = 330.

m = (ly2 - ly1) / (lx2-lx1)

b = -m * lx1 + ly1

for i in range(1, 7):
	img = cv2.imread(f'media/football/{i}.jpg')
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray, l, 2 * l)
	mask = cv2.dilate(edges, None, anchor=(-1, -1), iterations=2, borderType=1, borderValue=1)

	totalLabels, label_ids, values, _ = cv2.connectedComponentsWithStats(mask,
																		 4,
																		 cv2.CV_32S)
	output = np.zeros(gray.shape, dtype='uint8')

	for i in range(1, totalLabels):

		area = values[i, cv2.CC_STAT_AREA]
		if (area > 160) and (area < 300):
			componentMask = (label_ids == i).astype("uint8") * 255
			output = cv2.bitwise_or(output, componentMask)

	circles = cv2.HoughCircles(output, cv2.HOUGH_GRADIENT, 1, 10, param1=2 * l, param2=5, maxRadius=9)

	if circles is not None:
		circles = circles[0].astype(np.uint32)

		for circle in circles:
			cv2.circle(gray, (circle[0], circle[1]), circle[2], 255, 2)

			yhat = circle[0] * m + b

			if yhat > circle[1]:
				goal = False
			else:
				distance = abs(m * circle[0] - circle[1] + b) / (m ** 2 + 1) ** 0.5
				print(distance, circle[2])
				if distance > circle[2]:
					goal = True
					break

			x1 = circle[0] - circle[2] / 2
			x2 = circle[0] + circle[2] / 2
			y1 = circle[1] - circle[2] / 2
			y2 = circle[1] + circle[2] / 2

			cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
		if goal:
			break

	cv2.imshow('Frames', img)

	cv2.waitKey(0)

	if goal:
		break

if goal:
	print('GOAAALL!')
cv2.destroyAllWindows()