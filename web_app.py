import os
import cv2
import pandas as pd
import PIL.Image as Image
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from sklearn.neighbors import KernelDensity
from pathlib import Path
from ultralytics import ASSETS, YOLO
from sklearn.model_selection import GridSearchCV

DIR_NAME = Path(os.path.dirname(__file__))
DETECTION_MODEL_n = os.path.join(DIR_NAME, 'models', 'YOLOv8-N_CNO_Detection.pt')
DETECTION_MODEL_s = os.path.join(DIR_NAME, 'models', 'YOLOv8-S_CNO_Detection.pt')
DETECTION_MODEL_m = os.path.join(DIR_NAME, 'models', 'YOLOv8-M_CNO_Detection.pt')
DETECTION_MODEL_l = os.path.join(DIR_NAME, 'models', 'YOLOv8-L_CNO_Detection.pt')
DETECTION_MODEL_x = os.path.join(DIR_NAME, 'models', 'YOLOv8-X_CNO_Detection.pt')

# MODEL = os.path.join(DIR_NAME, 'models', 'YOLOv8-M_CNO_Detection.pt')
# model = YOLO(MODEL)
# cno_df = pd.DataFrame()


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

    cno_count = []
    cno_col = []
    afm_image = []
    cno_image = []
    kde_image = []
    file_name = []

    total_layer_area = []
    total_layer_cno = []
    total_layer_density = []
    avg_area_col = []
    total_area_col = []

    for idx, result in enumerate(results):
        cno = len(result.boxes)

        file_label = img[idx].split(os.sep)[-1]
        single_layer_area = []
        single_layer_cno = []
        single_layer_density = []
        total_area = 0
        if cno < 5:
            avg_area_col.append(np.nan)
            total_area_col.append(np.nan)
            nan_arr = np.empty([25])
            nan_arr[:] = np.nan
            total_layer_area.append(nan_arr)
            total_layer_cno.append(nan_arr)
            total_layer_density.append(nan_arr)
        else:
            cno_coor = np.empty([cno, 2], dtype=int)

            for j in range(cno):
                # w = r.boxes.xywh[j][2]
                # h = r.boxes.xywh[j][3]
                # area = (math.pi * w * h / 4) * 20 * 20 / (512 * 512)
                # total_area += area
                # bbox_img = r.orig_img
                x = round(result.boxes.xywh[j][0].item())
                y = round(result.boxes.xywh[j][1].item())

                x1 = round(result.boxes.xyxy[j][0].item())
                y1 = round(result.boxes.xyxy[j][1].item())
                x2 = round(result.boxes.xyxy[j][2].item())
                y2 = round(result.boxes.xyxy[j][3].item())

                cno_coor[j] = [x, y]
                cv2.rectangle(result.orig_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            im_array = result.orig_img
            afm_image.append([img[idx], file_label])
            cno_image.append([Image.fromarray(im_array[..., ::-1]), file_label])
            cno_count.append(cno)
            file_name.append(file_label)

            ### ============================
            """
            kde = KernelDensity(metric='euclidean', kernel='gaussian', algorithm='ball_tree')

            # Finding Optimal Bandwidth
            ti = time.time()
            if cno < 7:
                fold = cno
            else:
                fold = 7
            gs = GridSearchCV(kde, {'bandwidth': np.linspace(20, 60, 41)}, cv=fold)
            cv = gs.fit(cno_coor)
            bw = cv.best_params_['bandwidth']
            tf = time.time()
            print("Finding optimal bandwidth={:.2f} ({:n}-fold cross-validation): {:.2f} secs".format(bw, cv.cv,
                                                                                                      (tf - ti)))
            kde.bandwidth = bw
            _ = kde.fit(cno_coor)

            xgrid = np.arange(0, result.orig_img.shape[1], 1)
            ygrid = np.arange(0, result.orig_img.shape[0], 1)
            xv, yv = np.meshgrid(xgrid, ygrid)
            xys = np.vstack([xv.ravel(), yv.ravel()]).T
            gdim = xv.shape
            zi = np.arange(xys.shape[0])
            zXY = xys
            z = np.exp(kde.score_samples(zXY))
            zg = -9999 + np.zeros(xys.shape[0])
            zg[zi] = z

            xyz = np.hstack((xys[:, :2], zg[:, None]))
            x = xyz[:, 0].reshape(gdim)
            y = xyz[:, 1].reshape(gdim)
            z = xyz[:, 2].reshape(gdim)
            levels = np.linspace(0, z.max(), 26)
            print("levels", levels)

            for j in range(len(levels) - 1):
                area = np.argwhere(z >= levels[j])
                area_concatenate = numcat(area)
                CNO_concatenate = numcat(cno_coor)
                ecno = np.count_nonzero(np.isin(area_concatenate, CNO_concatenate))
                layer_area = area.shape[0]
                if layer_area == 0:
                    density = np.round(0.0, 4)
                else:
                    density = np.round((ecno / layer_area) * 512 * 512 / 400, 4)
                print("Level {}: Area={}, CNO={}, density={}".format(j, layer_area, ecno, density))
                single_layer_area.append(layer_area)
                single_layer_cno.append(ecno)
                single_layer_density.append(density)

            total_layer_area.append(single_layer_area)
            total_layer_cno.append(single_layer_cno)
            total_layer_density.append(single_layer_density)

            
            # Plot CNO Distribution
            plt.contourf(x, y, z, levels=levels, cmap=plt.cm.bone)
            plt.axis('off')
            # plt.gcf().set_size_inches(8, 8)
            plt.gcf().set_size_inches(8 * (gdim[1] / gdim[0]), 8)
            plt.gca().invert_yaxis()
            plt.xlim(0, gdim[1] - 1)
            plt.ylim(gdim[0] - 1, 0)
            kde_image.append([plt.figure(), file_label])
            #plt.savefig(os.path.join(kde_dir, '{}_{}_{}_KDE.png'.format(file_list[idx], model_type, conf)),
            #            bbox_inches='tight', pad_inches=0)
            """










        ### ============================

    data = {
        "Files": file_name,
        "CNO Count": cno_count,
    }

    # load data into a DataFrame object:
    cno_df = pd.DataFrame(data)

    return cno_df, afm_image, cno_image, kde_image


def numcat(arr):
    arr_size = arr.shape[0]
    arr_cat = np.empty([arr_size, 1], dtype=np.int32)
    for i in range(arr.shape[0]):
        arr_cat[i] = arr[i][0] * 1000 + arr[i][1]
    return arr_cat


def highlight_max(s, props=''):
    return np.where(s == np.nanmax(s.values), props, '')


def highlight_df(df, data: gr.SelectData):

    styler = df.style.apply(lambda x: ['background: lightgreen'
                               if x.Files == data.value["caption"]
                               else None for i in x], axis=1)

    # print("selected", data.value["caption"])
    return data.value["caption"], styler

def reset():
    name_textbox = ""
    gender_radio = None
    age_slider = 0
    fitzpatrick = 1
    history = []
    model_radio = "YOLOv8-M"
    input_files = []
    conf_slider = 0.2
    iou_slider = 0.5
    analysis_results = []
    afm_gallery = []
    cno_gallery = []
    test_label = ""

    return name_textbox, gender_radio, age_slider, fitzpatrick, history, model_radio, input_files, conf_slider, \
        iou_slider, analysis_results, afm_gallery, cno_gallery, test_label


with gr.Blocks(title="AFM AI Analysis", theme="default") as app:
    with gr.Row():
        with gr.Column():
            # gr.Markdown("User Information")
            with gr.Accordion("User Information", open=True):
                name_textbox = gr.Textbox(label="Name")
                with gr.Row():
                    gender_radio = gr.Radio(["Male", "Female"], label="Gender", interactive=True, scale=1)
                    age_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Age", interactive=True, scale=2)
                with gr.Group():
                    fitzpatrick = gr.Slider(minimum=1, maximum=6, step=1, value=1, label="Fitzpatrick", interactive=True)
                history = gr.Checkboxgroup(["Familial Disease", "Allergic Rhinitis", "Asthma"], label="Medical History", interactive=True)

            input_files = gr.File(file_types=["image"], file_count="multiple", label="Upload Image")
            # gr.Markdown("Model Configuration")
            with gr.Accordion("Model Configuration", open=False):
                model_radio = gr.Radio(["YOLOv8-N", "YOLOv8-S", "YOLOv8-M", "YOLOv8-L", "YOLOv8-X"], label="Model Selection", value="YOLOv8-M")
                conf_slider = gr.Slider(minimum=0, maximum=1, value=0.2, label="Confidence threshold")
                iou_slider = gr.Slider(minimum=0, maximum=1, value=0.5, label="IoU threshold")
            with gr.Row():
                analyze_btn = gr.Button("Analyze")
                clear_btn = gr.Button("Reset")
        with gr.Column():
            analysis_results = gr.Dataframe(headers=["Files", "CNO Count"], interactive=False)
            # cno_label = gr.Label(label="Analysis Results")
            with gr.Tab("AFM"):
                afm_gallery = gr.Gallery(label="Result", show_label=True, columns=3, object_fit="contain")
            with gr.Tab("CNO"):
                cno_gallery = gr.Gallery(label="Result", show_label=True, columns=3, object_fit="contain")
            with gr.Tab("KDE"):
                kde_gallery = gr.Gallery(label="Result", show_label=True, columns=3, object_fit="contain")
            test_label = gr.Label(label="Analysis Results")
            # cno_img = gr.Image(type="pil", label="Result")

    analyze_btn.click(
        fn=predict_image,
        inputs=[name_textbox, model_radio, input_files, conf_slider, iou_slider],
        outputs=[analysis_results, afm_gallery, cno_gallery, kde_gallery]
    )

    clear_btn.click(reset, outputs=[name_textbox, gender_radio, age_slider, fitzpatrick, history, model_radio,
                                    input_files, conf_slider, iou_slider, analysis_results, afm_gallery, cno_gallery,
                                    test_label])

    afm_gallery.select(highlight_df, inputs=analysis_results, outputs=[test_label, analysis_results])
    cno_gallery.select(highlight_df, inputs=analysis_results, outputs=[test_label, analysis_results])


"""
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
"""

if __name__ == '__main__':
    # iface.launch()
    app.launch(share=False, auth=[('jenhw', 'admin'), ('user', 'admin')], auth_message="Enter your username and password")
    # app.launch(share=True)
