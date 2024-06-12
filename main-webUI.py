
# import PIL.Image as Image
import gradio as gr
from ultralytics import YOLO
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from paddleocr import PaddleOCR
import numpy as np
import cv2

ocr_recognition = pipeline(Tasks.ocr_recognition, model='/home/binbin/.cache/modelscope/hub/damo/cv_convnextTiny_ocr-recognition-licenseplate_damo')
model = YOLO('/home/binbin/deeplearning/yolov8/yolov8_paddleOCR/model/license_plate.pt')#license_plate_detector
PP_ocr = PaddleOCR(use_angle_cls=True, lang="ch")
def predict_image(img,m):

    license_plates = model(img)[0]
    print('-----------------------------')
    print(license_plates)
    print('-----------------------------')
    license_plate_coordinates = []
    highest_score = 0
    highest_plate_coords = (0, 0, 0, 0)
    # 遍历所有检测到的车牌
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        # 将车牌的坐标和得分添加到列表中
        license_plate_coordinates.append((int(x1), int(y1), int(x2), int(y2), score))

        if score > highest_score:
            highest_score = score
            highest_plate_coords = (int(x1), int(y1), int(x2), int(y2))
    print(highest_plate_coords)
    # cropped_img = img.crop(highest_plate_coords)
    cropped_img = img[highest_plate_coords[1]:highest_plate_coords[3], highest_plate_coords[0]:highest_plate_coords[2]]
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    # 使用拉普拉斯算子进行锐化
    alpha = m
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = np.uint8(np.absolute(lap) * alpha + gray * (1 - alpha))

    out_text = ocr_recognition(lap)
    out_text2 = PP_ocr.ocr(lap)
    print(out_text2)
    
    if out_text2[0] is not None :

        for i in range(len(out_text2[0])):
            print(out_text2[0][i][1][0])
            out_text_pp = out_text2[0][i][1][0]
        print(out_text_pp)
        return lap , out_text['text'][0],out_text2[0][0][1][0]
    else:
        out_text2 = PP_ocr.ocr(img)
        for i in range(len(out_text2[0])):
            print(out_text2[0][i][1][0])
            out_text_pp = out_text2[0][i][1][0]
        print(out_text_pp)
        return img , out_text['text'][0],out_text_pp
    # return lap , out_text['text'][0],out_text2[0][0][1][0]


interface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="numpy", label="Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.10, label="锐化")
    ],
    outputs=[
        gr.Image(type="numpy", label="Result"),
        gr.Textbox(label='车牌-读光'),
        gr.Textbox(label='车牌-Paddleocr')
    ],
    title="基于YoLov8的车牌识别系统",
    description="来试试吧！"
)
interface.queue()
interface.launch(share=True)
