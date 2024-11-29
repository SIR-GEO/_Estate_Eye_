from ultralytics import YOLO
import torch
from paddleocr import PaddleOCR
import paddle
import importlib

def initialize_model():
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device = 0
        torch.cuda.set_device(device)
        
        # Check if paddlepaddle-gpu is installed
        paddle_gpu_available = False
        try:
            if importlib.util.find_spec("paddlepaddle_gpu"):
                paddle_gpu_available = True
                paddle.device.set_device('gpu')
        except:
            paddle.device.set_device('cpu')
            print("Warning: PaddlePaddle GPU not available, using CPU for OCR")
    else:
        device = "cpu"
        paddle.device.set_device('cpu')

    # Initialize YOLO with specific device
    model = YOLO("yolov8n.pt")
    if cuda_available:
        model.to(device)

    # Initialize PaddleOCR with GPU if available
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='en',
        use_gpu=cuda_available and paddle_gpu_available,
        gpu_mem=4000
    )

    return model, device, ocr, cuda_available