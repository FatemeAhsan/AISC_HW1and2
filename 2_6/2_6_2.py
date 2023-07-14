# In the name of Allah
import cv2
from ultralytics import YOLO
import numpy as np
import math

l = 125

model = YOLO('models/yolov8x')

for i in range(1, 7):
	img = cv2.imread(f'media/football/{i}.jpg')

	results = model.predict(img)

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray, l, 2 * l)
	lines = cv2.HoughLines(edges, 1, np.pi / 180, 220, None, 0, 0)

	for result in results:
		for (obj_xyxy, obj_cls) in zip(result.boxes.xyxy, result.boxes.cls):
			obj_cls = int(obj_cls)

			if obj_cls == 32:
				x1 = obj_xyxy[0].item()
				y1 = obj_xyxy[1].item()
				x2 = obj_xyxy[2].item()
				y2 = obj_xyxy[3].item()

				ball_x = (x2 + x1) / 2
				ball_y = (y2 + y1) / 2

				ball_r = (x2 - x1 + y2 - y1) / 4

				if lines is not None:
					for i in range(0, len(lines)):
						rho = lines[i][0][0]
						theta = lines[i][0][1]
						if rho < 0:
							a = math.cos(theta)
							b = math.sin(theta)
							x0 = a * rho
							y0 = b * rho
							lx1 = int(x0 + 1000 * -b)
							ly1 = int(y0 + 1000 * a)
							lx2 = int(x0 - 1000 * -b)
							ly2 = int(y0 - 1000 * a)

							m = (ly2 - ly1) / (lx2 - lx1)

							b = -m * lx1 + ly1

							yhat = ball_x * m + b
							if yhat > ball_y:
								goal = False
							else:
								distance = abs(m * ball_x - ball_y + b) / (m ** 2 + 1) ** 0.5
								print(distance, ball_r)
								if distance > ball_r:
									goal = True
									break

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
