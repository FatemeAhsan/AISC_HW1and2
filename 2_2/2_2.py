# In the name of Allah
import cv2
import time

try:
	import vlc
except:
    import os
    os.add_dll_directory('D:\Programs\VLC')
    import vlc

class FClass():
	def __init__(self):
		self.face_cascade = cv2.CascadeClassifier('files/haarcascade_frontalface_default.xml')
		self.eye_cascade = cv2.CascadeClassifier('files/haarcascade_eye.xml')
		self.cap = cv2.VideoCapture(0)


	def f_method(self):
		inp = input()
		
		if inp == 'music':
			vlc.MediaPlayer('files/anewbeginning.mp3').play()
			time.sleep(15)
		elif inp == 'movie':
			cv2.imshow('Movie Cover', cv2.imread('files/In the Heart of the Sea.jpg'))
		elif inp == 'book':
			print('Deep Learning with Python')

	def stream_webcam(self):
		while self.cap.isOpened():
			ret, frame = self.cap.read()

			if ret:
				frame = cv2.flip(frame, 1)

				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

				faces = self.face_cascade.detectMultiScale(gray)

				for (x, y, w, h) in faces:
					cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
					face_gray = gray[y:y + h, x:x + w]
					face_color = frame[y:y + h, x:x + w]
					
					eyes = self.eye_cascade.detectMultiScale(face_gray)
					for (ex, ey, ew, eh) in eyes:
						cv2.rectangle(face_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    
					
				cv2.imshow('Webcam', frame)

				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
			else:
				break

		self.cap.release()
		cv2.destroyAllWindows()

fc = FClass()

fc.f_method()

fc.stream_webcam()
