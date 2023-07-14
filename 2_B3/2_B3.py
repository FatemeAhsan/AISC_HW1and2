# In the name of Allah
import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO('models/yolov8x-seg')

img = cv2.imread('media/sample.jpg')

results = model(img, classes=0, retina_masks=True)

data = results[0].masks[0].data[0]
rs = np.zeros((data.shape[0], data.shape[1], 3))
img_resized = cv2.resize(img, (rs.shape[1], rs.shape[0]))

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if data[i, j] > 0:
            rs[i, j] = img_resized[i, j]
            
cv2.waitKey(0)    

rs = rs.astype(np.uint8)

tmp = cv2.cvtColor(rs, cv2.COLOR_BGR2GRAY)
  
_, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
  
b, g, r = cv2.split(rs)
  
rs = cv2.merge([b, g, r, alpha], 4)

rs = cv2.resize(rs, (img.shape[1], img.shape[0]))

cv2.imwrite('out.png', rs)

cv2.destroyAllWindows()
