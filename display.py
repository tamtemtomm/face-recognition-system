import customtkinter as ctk, queue, cv2, time
from PIL import Image
from tools.tools import annotate_frame
from tools.db import MongoDatabase

import yaml

with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)

db = MongoDatabase(uri=CONFIG['mongo_uri'])

class FaceDetectionApp(ctk.CTk):
    def __init__(self, 
                 det_queue : queue.Queue, 
                 stop_event : queue.Queue):
        
        super().__init__()

        self.title("Face Detection UI")
        self.geometry("1280x600")
        self.det_queue = det_queue
        self.stop_event = stop_event
        self.seen_names = set() 

        # UI layout
        self.columnconfigure((0, 1), weight=1)
        self.rowconfigure(0, weight=1)

        # Left panel - camera feed
        self.left_panel = ctk.CTkLabel(self, text="", width=640, height=480)
        self.left_panel.grid(row=0, column=0, padx=10, pady=10)

        # Right panel - placeholder
        self.info_frame = ctk.CTkScrollableFrame(self, label_text="Absence Infomation")
        self.info_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.info_frame.columnconfigure(0, weight=1)
        
        # # Header row for info_frame
        # header_frame = ctk.CTkFrame(self.info_frame)
        # header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(0, 10))
        # header_frame.columnconfigure((0, 1, 2), weight=1)

        # timestamp_header = ctk.CTkLabel(header_frame, text="Timestamp", font=("Arial", 14, "bold"))
        # timestamp_header.grid(row=0, column=0, padx=10, pady=2, sticky="w")

        # id_header = ctk.CTkLabel(header_frame, text="ID", font=("Arial", 14, "bold"))
        # id_header.grid(row=0, column=1, padx=10, pady=2, sticky="w")

        # name_header = ctk.CTkLabel(header_frame, text="Name", font=("Arial", 14, "bold"))
        # name_header.grid(row=0, column=2, padx=10, pady=2, sticky="w")

        # # Track next row index for detection blocks
        # self.next_info_row = 1

        self.update_image_loop()

    def update_image_loop(self):
        if not self.stop_event.is_set():
            try:
                frame, results = self.det_queue.get_nowait()
                annotated = annotate_frame(frame=frame, results=results)

                # Convert BGR to RGB
                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(annotated)
                img = img.resize((640, 480))
                # Convert PIL Image to CTkImage
                ctk_img = ctk.CTkImage(light_image=img, size=(640, 480))
                
                self.left_panel.configure(image=ctk_img, text="")
                self.left_panel.image = ctk_img
                
                if isinstance(results, list):
                    for result in results:
                        class_name = result.get('class')
                        if class_name and class_name != "UNKNOWN" and class_name not in self.seen_names:
                            self.add_detection_block(result)
                            self.seen_names.add(result.get('class'))
                            db.insert_one(result)

            except queue.Empty:
                pass

            self.after(15, self.update_image_loop)
        else:
            self.destroy()
    
    def add_detection_block(self, result:dict):

        block = ctk.CTkFrame(self.info_frame)
        block.grid(sticky="ew", padx=10, pady=5)
        block.columnconfigure((0, 1, 2), weight=1)

        # Timestamp Label
        timestamp_label = ctk.CTkLabel(block, text=f"{result['timestamp'].strftime("%Y-%m-%d %H:%M:%S")}", font=("Arial", 14, "bold"))
        timestamp_label.grid(row=0, column=0, padx=10, pady=2, sticky="w")

        # ID Label
        id_label = ctk.CTkLabel(block, text=f"{result['id']}", font=("Arial", 14, "bold"))
        id_label.grid(row=0, column=1, padx=10, pady=2, sticky="w")

        # Name Label
        name_label = ctk.CTkLabel(block, text=f"{result['name']}", font=("Arial", 14, "bold"))
        name_label.grid(row=0, column=2, padx=10, pady=2, sticky="w")