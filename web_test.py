import os
import cv2
import PIL.Image as Image
import gradio as gr
import numpy as np
import math
from pathlib import Path
from ultralytics import ASSETS, YOLO

DIR_NAME = Path(os.path.dirname(__file__))
DETECTION_MODEL_n = os.path.join(DIR_NAME, 'models', 'YOLOv8-N_CNO_Detection.pt')
DETECTION_MODEL_s = os.path.join(DIR_NAME, 'models', 'YOLOv8-S_CNO_Detection.pt')
DETECTION_MODEL_m = os.path.join(DIR_NAME, 'models', 'YOLOv8-M_CNO_Detection.pt')
DETECTION_MODEL_l = os.path.join(DIR_NAME, 'models', 'YOLOv8-L_CNO_Detection.pt')
DETECTION_MODEL_x = os.path.join(DIR_NAME, 'models', 'YOLOv8-X_CNO_Detection.pt')

# MODEL = os.path.join(DIR_NAME, 'models', 'YOLOv8-M_CNO_Detection.pt')
# model = YOLO(MODEL)


def predict_image(name, model, img, conf_threshold, iou_threshold):
    """Predicts and plots labeled objects in an image using YOLOv8 model with adjustable confidence and IOU thresholds."""
    gr.Info("Starting process")
    # gr.Warning("Name is empty")
    if name == "":
        gr.Warning("Name is empty")

    if model == 'YOLOv8-N':
        CNO_model = YOLO(DETECTION_MODEL_n)
    elif model == 'YOLOv8-S':
        CNO_model = YOLO(DETECTION_MODEL_s)
    elif model == 'YOLOv8-M':
        CNO_model = YOLO(DETECTION_MODEL_m)
    elif model == 'YOLOv8-L':
        CNO_model = YOLO(DETECTION_MODEL_l)
    else:
        CNO_model = YOLO(DETECTION_MODEL_x)

    results = CNO_model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=False,
        show_conf=False,
        imgsz=512,
        max_det=1200
    )

    for r in results:
        CNO = len(r.boxes)
        CNO_coor = np.empty([CNO, 2], dtype=int)
        for j in range(CNO):
            # w = r.boxes.xywh[j][2]
            # h = r.boxes.xywh[j][3]
            # area = (math.pi * w * h / 4) * 20 * 20 / (512 * 512)
            # total_area += area
            # bbox_img = r.orig_img
            x = round(r.boxes.xywh[j][0].item())
            y = round(r.boxes.xywh[j][1].item())

            x1 = round(r.boxes.xyxy[j][0].item())
            y1 = round(r.boxes.xyxy[j][1].item())
            x2 = round(r.boxes.xyxy[j][2].item())
            y2 = round(r.boxes.xyxy[j][3].item())

            CNO_coor[j] = [x, y]
            cv2.rectangle(r.orig_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        im_array = r.orig_img
        im = Image.fromarray(im_array[..., ::-1])

    CNO_count = "CNO Count: " + str(CNO)

    return CNO_count, im


iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Textbox(label="User Name"),
        gr.Radio(["YOLOv8-N", "YOLOv8-S", "YOLOv8-M", "YOLOv8-L", "YOLOv8-X"], value="YOLOv8-M"),
        # gr.Image(type="filepath", label="Upload Image"),
        gr.File(file_types=["image"], file_count="multiple", label="Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.2, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.5, label="IoU threshold")
    ],
    outputs=[gr.Label(label="Analysis Results"), gr.Image(type="pil", label="Result")],
    title="AFM AI Analysis",
    description="Upload images for inference. The YOLOv8-M model is used by default.",
    theme=gr.themes.Default()
)

if __name__ == '__main__':
    iface.launch()
