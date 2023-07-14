# In the name of Allah
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import math

model = YOLO('models/yolov8x')

video = cv2.VideoCapture('media/video_2023-04-23_05-19-13.mp4')
mask = cv2.imread('media/mask.jpg')

car_tracker = DeepSort(max_age=5, n_init=2)
motorcycle_tracker = DeepSort(max_age=5, n_init=2)

n_cars = 0
n_motorcycles = 0

limits = [330, 420, 800, 420]
offset = 20

car_added_ids = []
motorcycle_added_ids = []


def show_inc_track(tracks, added_ids):
    cnt = 0
    for track in tracks:
        x1, y1, x2, y2 = track.to_ltrb()
        cv2.rectangle(frame_masked, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
        y = int((y1 + y2) / 2)
        if limits[1] - offset < y < limits[1] + offset and track.track_id not in added_ids:
            added_ids.append(track.track_id)
            cnt += 1
    return cnt


def add_track(box):
    global tracker
    x1, y1, x2, y2 = box.xyxy[0]
    conf = math.ceil(box.conf[0] * 100) / 100
    return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)], conf, 'vehicle'


while video.isOpened():
    ret, frame = video.read()
    frame_masked = cv2.bitwise_and(frame, mask)

    if ret:
        results = model.predict(frame_masked, stream=True)

        car_detections = []
        motorcycle_detections = []

        for result in results:
            for box in result.boxes:
                obj_cls = int(box.cls[0])
                if obj_cls == 2:
                    car_detections.append(add_track(box))
                elif obj_cls == 3:
                    motorcycle_detections.append(add_track(box))
        print(car_added_ids)

        car_tracks = car_tracker.update_tracks(car_detections, frame=frame_masked)
        motorcycle_tracks = motorcycle_tracker.update_tracks(motorcycle_detections, frame=frame_masked)
        n_cars += show_inc_track(car_tracks, car_added_ids)
        n_motorcycles += show_inc_track(motorcycle_tracks, motorcycle_added_ids)

        cv2.line(frame_masked, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 2)
        cv2.imshow('Frames', frame_masked)

        print(f'# cars: {n_cars}, # motorcycles: {n_motorcycles}')

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

print(f'# cars: {n_cars}, # motorcycles: {n_motorcycles}')

cv2.destroyAllWindows()
