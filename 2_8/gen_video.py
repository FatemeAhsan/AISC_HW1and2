import os
import numpy as np
import cv2
import mediapipe as mp


INP_PATH = 'media'

EYES_LM_INDXS = [33, 263]

MAX_EYES_WIDTH_SIZE = 424
PAD_SIZE = (3000, 3000, 3)
RS_SIZE = (4000, 4000, 3)
FINAL_SIZE = (2000, 2000)

VIDEO_FILE_NAME = 'vd.mp4'
DESIRED_FPS = 4

AUDIO_FILENAME = 'anewbeginning.mp3'
RS_FILENAME = 'rs.mp4'

video_wr = cv2.VideoWriter(VIDEO_FILE_NAME, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), DESIRED_FPS, FINAL_SIZE)  

face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True,
                                            min_tracking_confidence=0.8, min_detection_confidence=0.7)

for file in os.listdir(INP_PATH):
    name, ext = os.path.splitext(file)
    print(name)
    if ext == '.mp3':
        continue
    
    img = cv2.cvtColor(cv2.imread(os.path.join(INP_PATH, file)), cv2.COLOR_BGR2RGB)
    
    results = face_mesh.process(img)

    width, height = img.shape[1], img.shape[0]

    if results.multi_face_landmarks is not None:
        for face_landmarks in results.multi_face_landmarks:
            mesh_points = np.array(
                [np.multiply([p.x, p.y], [width, height]).astype(int) for p in face_landmarks.landmark])
            L_EYE = mesh_points[EYES_LM_INDXS[0]]
            R_EYE = mesh_points[EYES_LM_INDXS[1]]

            eyes_x_min = L_EYE[0]
            eyes_x_max = R_EYE[0]
            
            width = int(MAX_EYES_WIDTH_SIZE / (eyes_x_max - eyes_x_min) * img.shape[1])
            height = int(MAX_EYES_WIDTH_SIZE / (eyes_x_max - eyes_x_min) * img.shape[0])
            image_resized = cv2.resize(img, (width, height))

            R_EYE = (MAX_EYES_WIDTH_SIZE / (eyes_x_max - eyes_x_min) * R_EYE).astype(int)
            L_EYE = (MAX_EYES_WIDTH_SIZE / (eyes_x_max - eyes_x_min) * L_EYE).astype(int)
            
            rad = np.arctan2((L_EYE[1] - R_EYE[1]), MAX_EYES_WIDTH_SIZE)
            pv = (R_EYE + L_EYE + np.flip(PAD_SIZE[:2]) - np.flip(image_resized.shape[:2])) // 2
            
            rot_mat = cv2.getRotationMatrix2D((int(pv[0]), int(pv[1])), -rad * 180 / np.pi, 1.0)

            pad = np.full(PAD_SIZE, 255, dtype=np.uint8)
            
            pad[(PAD_SIZE[0] - image_resized.shape[0]) // 2:(PAD_SIZE[0] + image_resized.shape[0]) // 2, (PAD_SIZE[1] - image_resized.shape[1]) // 2:(PAD_SIZE[1] + image_resized.shape[1]) // 2] = image_resized

            image = cv2.warpAffine(pad, rot_mat, (pad.shape[1], pad.shape[0]), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
            
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
            rs = np.full(RS_SIZE, 255, dtype=np.uint8)
            
            rs[RS_SIZE[0] // 2 - pv[1]:RS_SIZE[0] // 2 - pv[1] + image.shape[0], RS_SIZE[1] // 2 - pv[0]:RS_SIZE[0] // 2 - pv[0] + image.shape[1]] = image

            rs = cv2.resize(rs, FINAL_SIZE).astype(np.uint8)
            video_wr.write(rs)

video_wr.release()
cv2.destroyAllWindows()

os.system(f'ffmpeg -i {VIDEO_FILE_NAME} -i {os.path.join(INP_PATH, AUDIO_FILENAME)} -t 00:00:05 -c copy -y {RS_FILENAME}')