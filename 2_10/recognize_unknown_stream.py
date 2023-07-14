import face_recognition
import cv2
import pickle

src = cv2.VideoCapture(0)

with open('known_faces.pickle', 'rb') as f:
	names, encodings = pickle.load(f)

while src.isOpened():
	ret, frame = src.read()

	if ret:
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		locs = face_recognition.face_locations(rgb, model='hog')
		encods = face_recognition.face_encodings(rgb, locs)

		for (loc, encod) in zip(locs, encods):
			top_left = loc[3], loc[0]
			bottom_right = loc[1], loc[2]

			results = face_recognition.compare_faces(encodings, encod, 0.6)

			for (i, res) in enumerate(results):
				if res:
					new_name = names[i]
					cv2.putText(frame, new_name, top_left,
						cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 1)
					break

			cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)

		cv2.imshow('Webcam', frame)

	q = cv2.waitKey(1)
	if q == ord('q') or q == ord('Q'):
		break

cv2.destroyAllWindows()

