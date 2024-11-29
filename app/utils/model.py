from ultralytics import YOLO
import torch
from paddleocr import PaddleOCR

def initialize_model():
    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8n.pt")
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=torch.cuda.is_available())
    return model, device, ocr