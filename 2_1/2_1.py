# In the name of Allah
import cv2
import numpy as np
import time

name = 'Fateme Ahsan'

cap = cv2.VideoCapture(0)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

t0 = time.time()

while cap.isOpened():
    ret, frame = cap.read()

    t_now = time.time() - t0

    if ret:

        frame = cv2.resize(cv2.flip(frame, 1), None, fx=0.5, fy=0.5)
        
        red = frame.copy()
        red[:, :, 2] = 255
        
        inv = 255 - frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        out_frame = np.concatenate((np.concatenate((frame, red), 1), np.concatenate((inv, np.stack((gray,) * 3, -1)), 1)), 0)
                    
 
        cv2.putText(out_frame, f'name: {name}', (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)
        cv2.putText(out_frame, f'time: {round(t_now, 2)}', (100, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 255), 2)
        
        cv2.imshow('Demo', out_frame)

    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
cap.release()
