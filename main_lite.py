import cv2, torch, copy, yaml
from models import initialize_arcface, initialize_mtcnn, BACKBONE
from tools import batching_detect_face, batching_predict_face, load_facebank, annotate_frame

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    with open("config.yaml") as f:
        CONFIG = yaml.safe_load(f)
    
    mtcnn = initialize_mtcnn(device, selection_method="probability", keep_all=True)
    arcface = initialize_arcface(device=device, bbone=BACKBONE.RESNET50)
    facebank = load_facebank(CONFIG["facebank_output"], device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    frame_counter = 0
    FRAME_SKIP = 10  # Detect every 5 frames
    last_result = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        temp_frame = copy.deepcopy(frame)

        if frame_counter % FRAME_SKIP == 0:
            result = batching_detect_face(temp_frame, mtcnn)
            last_result = batching_predict_face(temp_frame, result, arcface, facebank, device=device)

        if last_result is not None:
            temp_frame = annotate_frame(frame=temp_frame, results=last_result)

        frame_counter += 1

        cv2.imshow("Live Face Detection", temp_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
