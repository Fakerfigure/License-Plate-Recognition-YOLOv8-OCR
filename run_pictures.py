import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from paddleocr import PaddleOCR
from openpyxl import Workbook

# 假设你的模型加载和初始化代码在这里
ocr_recognition = pipeline(Tasks.ocr_recognition, model='/home/binbin/.cache/modelscope/hub/damo/cv_convnextTiny_ocr-recognition-licenseplate_damo')
model = YOLO('/home/binbin/deeplearning/yolov8/YOLOv8-license-plate-recognize/models/license_plate_detector.pt')#license_plate_detector
PP_ocr = PaddleOCR(use_angle_cls=True, lang="ch")

def predict_image(img_path, ocr_recognition, PP_ocr):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading image {img_path}. Skipping this image.")
        return None

    license_plates = model(img)[0]
    print('-----------------------------')
    print(license_plates)
    print('-----------------------------')
    
    # ... 省略中间处理步骤 ...
    highest_score = 0
    highest_plate_coords = None
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        if score > highest_score:
            highest_score = score
            highest_plate_coords = (int(x1), int(y1), int(x2), int(y2))

    if highest_plate_coords is None:
        print(f"No license plate detected in {img_path}. Skipping this image.")
        return None

    cropped_img = img[highest_plate_coords[1]:highest_plate_coords[3], highest_plate_coords[0]:highest_plate_coords[2]]
    out_text = ocr_recognition(cropped_img)
    print(out_text)
    out_text2 = PP_ocr.ocr(cropped_img)
    print('微调后:\n')
    print(out_text2)

    if out_text2[0] is not None:

        out_text_pp = out_text2[0][0][1][0]
        return img_path, out_text['text'][0], out_text_pp
    else:
        out_text2 = PP_ocr.ocr(img)
        if out_text2[0] is not None:
            for i in range(len(out_text2[0])):
                print(out_text2[0][i][1][0])
                out_text_pp = out_text2[0][i][1][0]
            return img_path, out_text['text'][0], out_text_pp
        else:
            out_text_pp = 'NULL'
            return img_path, out_text['text'][0], out_text_pp

def process_images_in_folder(folder_path, excel_path, ocr_recognition, PP_ocr):
    all_results = []
    for img_filename in os.listdir(folder_path):
        if img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, img_filename)
            result = predict_image(img_path, ocr_recognition, PP_ocr)
            if result is not None:
                # 使用os.path.basename()来获取不带路径的图片名称
                img_name = os.path.basename(img_path)
                all_results.append([img_name, result[1], result[2]])

    # 创建一个Excel工作簿
    wb = Workbook()
    ws = wb.active
    ws.append(['Image Name', 'Detected License Plate', 'Adjusted License Plate'])

    # 将结果写入Excel
    for res in all_results:
        ws.append(res)

    # 保存Excel文件
    wb.save(excel_path)
# 设置文件夹路径和Excel文件路径
# folder_path = '/home/binbin/Downloads/2024/base_test'
folder_path = '/home/binbin/Downloads/CLPD/CLPD_1200'
excel_path = '/home/binbin/Downloads/2024/save/results1200.xlsx'

# 运行脚本
process_images_in_folder(folder_path, excel_path, ocr_recognition, PP_ocr)