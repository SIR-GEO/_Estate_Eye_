import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import cv2
import numpy as np
from .utils.model import initialize_model
import time
import torch
from pyzbar.pyzbar import decode
from .utils.ai_utils import AIAnalyzer

# Get the absolute path to the static and templates directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "static")
templates_dir = os.path.join(BASE_DIR, "templates")

app = FastAPI()

# Mount static files with absolute path
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# Initialize model
model, device, ocr, cuda_available, barcode_decoder = initialize_model()

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Current CUDA Device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")
print(f"GPU Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"YOLO Device: {next(model.parameters()).device}")
print(f"PaddleOCR GPU Enabled: {cuda_available}")

# Add after other initializations
ai_analyzer = AIAnalyzer()

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        connection_active = True
        last_ocr_time = 0
        last_barcode_time = 0
        OCR_INTERVAL = 1
        BARCODE_INTERVAL = 1

        while connection_active:
            try:
                data = await websocket.receive_json()
                frame_data = bytes(data['frame'])
                detection_enabled = data.get('detection', False)
                ocr_enabled = data.get('ocr', False)
                barcode_enabled = data.get('barcode', False)

                if not frame_data:
                    continue

                nparr = np.frombuffer(frame_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                    
                processed_img = img.copy()

                # Barcode/QR Detection
                if barcode_enabled and (time.time() - last_barcode_time) >= BARCODE_INTERVAL:
                    barcodes = barcode_decoder(img)
                    last_barcode_time = time.time()
                    
                    if barcodes:
                        barcode_texts = []
                        for barcode in barcodes:
                            # Get the barcode polygon
                            points = barcode.polygon
                            if points is not None and len(points) > 0:
                                # Convert points to numpy array
                                pts = np.array(points, np.int32)
                                pts = pts.reshape((-1, 1, 2))
                                
                                # Draw the polygon
                                cv2.polylines(processed_img, [pts], True, (255, 0, 0), 2)
                                
                                # Add text above the barcode
                                text = f"{barcode.type}: {barcode.data.decode()}"
                                text_position = (points[0].x, points[0].y - 10)
                                cv2.putText(
                                    processed_img,
                                    text,
                                    text_position,
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 0, 0),
                                    2
                                )
                                barcode_texts.append(text)
                        
                        if barcode_texts:
                            await websocket.send_json({
                                "type": "barcode",
                                "texts": barcode_texts
                            })

                if ocr_enabled and (time.time() - last_ocr_time) >= OCR_INTERVAL:
                    ocr_results = ocr.ocr(img)
                    last_ocr_time = time.time()
                    
                    if ocr_results is not None and len(ocr_results) > 0:
                        texts = []
                        for line in ocr_results[0]:  # PaddleOCR returns a list of pages
                            if line is not None and len(line) >= 2:
                                box = line[0]
                                if box is not None and len(box) == 4:
                                    points = np.array(box).astype(np.int32)
                                    text = line[1][0]  # Just get the text, ignore confidence
                                    
                                    # Draw the box
                                    cv2.polylines(processed_img, [points], True, (0, 0, 255), 2)
                                    
                                    # Add text above the box without confidence
                                    text_position = (int(points[0][0]), int(points[0][1] - 10))
                                    cv2.putText(
                                        processed_img,
                                        text,
                                        text_position,
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        (0, 0, 255),
                                        2
                                    )
                                    texts.append(text)  # Only append the text

                        if texts:
                            await websocket.send_json({
                                "type": "ocr",
                                "texts": texts
                            })

                if detection_enabled:
                    results = model(img)
                    
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            label = model.names[int(box.cls[0])]
                            score = float(box.conf[0])

                            cv2.rectangle(processed_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(
                                processed_img,
                                f"{label} ({score:.2f})",
                                (x1, max(y1 - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2,
                            )

                _, buffer = cv2.imencode(".jpg", processed_img)
                await websocket.send_bytes(buffer.tobytes())

            except WebSocketDisconnect:
                connection_active = False
                break
            except Exception as e:
                print(f"Frame processing error: {str(e)}")
                continue

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    finally:
        print("WebSocket connection closed")

@app.post("/analyze_snapshot")
async def analyze_snapshot(request: Request):
    try:
        data = await request.json()
        image_data = np.frombuffer(bytes(data['frame']), np.uint8)
        img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        
        ocr_texts = data.get('ocr_texts', [])
        barcode_texts = data.get('barcode_texts', [])
        
        analysis = await ai_analyzer.analyze_snapshot(img, ocr_texts, barcode_texts)
        return {"analysis": analysis}
    except Exception as e:
        return {"error": str(e)}
