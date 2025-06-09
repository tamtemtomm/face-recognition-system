import cv2, queue, time, torch
from threading import  Event

from models import  BACKBONE, Backbone, IResNet100, MTCNN
from tools import batching_detect_face, batching_predict_face

device = "cuda" if torch.cuda.is_available() else "cpu"

def camera_thread(frame_queue: queue.Queue, stop_event : Event) -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        stop_event.set()
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass
        time.sleep(0.01)

    cap.release()

def detect_thread(
    frame_queue : queue.Queue,
    det_queue: queue.Queue,
    stop_event : Event, 
    mtcnn : MTCNN,
    arcface : Backbone | IResNet100,
    facebank : dict
    ) -> None:
    
    frame_counter = 0
    FRAME_SKIP = 10
    last_result = None
    last_frame = None

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        if frame_counter % FRAME_SKIP == 0:
            result = batching_detect_face(frame, mtcnn)
            result = batching_predict_face(frame, result, arcface, facebank, device=device)
            
            last_result = result
            last_frame = frame
        else:
            result = last_result
            last_frame = frame

        frame_counter += 1

        if result is not None and last_frame is not None:
            try:
                det_queue.put_nowait((last_frame, result))
            except queue.Full:
                pass