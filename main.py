import torch, yaml, queue
from threading import Thread, Event

from models import initialize_arcface, initialize_mtcnn, BACKBONE
from tools import load_facebank

from display import FaceDetectionApp
from process import camera_thread, detect_thread

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    with open("config.yaml") as f:
        CONFIG = yaml.safe_load(f)

    BBONE = BACKBONE.IRESNET100
    
    mtcnn = initialize_mtcnn(device, selection_method="probability", keep_all=True)
    arcface = initialize_arcface(device=device, bbone=BBONE)
    facebank = load_facebank(CONFIG["facebank_output"], device)

    frame_queue = queue.Queue(maxsize=3)
    det_queue = queue.Queue(maxsize=3)
    stop_event = Event()

    cam_thread = Thread(target=camera_thread, args=(frame_queue, stop_event))
    det_thread = Thread(target=detect_thread, args=(frame_queue, det_queue, stop_event, mtcnn, arcface, facebank))

    cam_thread.start()
    det_thread.start()

    app = FaceDetectionApp(det_queue, stop_event)
    try:
        app.mainloop()
    except KeyboardInterrupt:
        stop_event.set()

    cam_thread.join()
    det_thread.join()
    print("Stopped cleanly.")
