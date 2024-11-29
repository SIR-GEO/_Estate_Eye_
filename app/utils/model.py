from ultralytics import YOLO
import torch

def initialize_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8n.pt")
    return model, device