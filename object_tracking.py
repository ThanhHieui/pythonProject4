import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape


def create_square(canvas, color, x, y, width, height):
    canvas.create_rectangle(x, y, x + width, y + height, fill=color)


# Function to open video file from the local machine
def open_video_file():
    filename = filedialog.askopenfilename(title="Select Video File",
                                          filetypes=(("MP4 files", "*.mp4"),
                                                     ("AVI files", "*.avi"),
                                                     ("All files", "*.*")))
    return filename


def process_video(video_path, model, tracker, conf_threshold, tracking_class, class_names, colors):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1220, 800))

        results = model(frame)

        detect = []
        for detect_object in results.pred[0]:
            label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4]
            x1, y1, x2, y2 = map(int, bbox)
            class_id = int(label)

            if tracking_class is None:
                if confidence < conf_threshold:
                    continue
            else:
                if class_id != tracking_class or confidence < conf_threshold:
                    continue

            detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

        tracks = tracker.update_tracks(detect, frame=frame)

        for track in tracks:
            if track.is_confirmed():
                track_id = track.track_id

                ltrb = track.to_ltrb()
                class_id = track.get_det_class()
                x1, y1, x2, y2 = map(int, ltrb)
                color = colors[class_id]
                B, G, R = map(int, color)

                label = "{}-{}".format(class_names[class_id], track_id)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
                cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        yield frame

    cap.release()

def open_video_and_detect(root, canvas):
    video_path = open_video_file()
    if video_path:
        # Khởi tạo YOLOv9
        device = "cpu"  # "cpu": CPU, "mps:0"
        model = DetectMultiBackend(weights="weights/yolov9-c-converted.pt", device=device, fuse=True)
        model = AutoShape(model)

        tracker = DeepSort(max_age=30)
        conf_threshold = 0.5
        tracking_class = 0

        with open("data_ext/class.name") as f:
            class_names = f.read().strip().split('\n')

        colors = np.random.randint(0, 255, size=(len(class_names), 3))

        for frame in process_video(video_path, model, tracker, conf_threshold, tracking_class, class_names, colors):
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=img)
            canvas.create_image(305, 50, anchor=tk.NW, image=img)
            root.update()


def main():
    root = tk.Tk()
    root.title("Form Giao Diện")
    root.attributes('-fullscreen', True)

    square_width = 1220
    square_height = 800

    square1_x = 10
    square1_y = 50
    square2_x = 305
    square2_y = 50

    canvas = tk.Canvas(root, width=root.winfo_screenwidth(), height=root.winfo_screenheight())
    canvas.pack(fill=tk.BOTH, expand=True)

    open_file_button = tk.Button(root, text="Open File", command=lambda: open_video_and_detect(root, canvas))
    open_file_button.place(x=100, y=100)
    open_file_button.config(width=15, height=3, bg='white')

    close_button = tk.Button(root, text="Close", command=root.destroy)
    close_button.place(x=10, y=10)
    close_button.config(font=('Arial', 14), width=26, bg='yellow')

    create_square(canvas, "grey", square1_x, square1_y, square_width, square_height)
    create_square(canvas, "white", square2_x, square2_y, square_width, square_height)

    root.mainloop()


if __name__ == "__main__":
    main()
