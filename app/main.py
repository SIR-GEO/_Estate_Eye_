import sys
import os
import base64
import cv2
import numpy as np
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from .utils.model import initialize_model
import time
import torch
from pyzbar.pyzbar import decode
from .utils.ai_utils import AIAnalyzer, scrape_url
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse
from fastapi.responses import Response

# Get the absolute path to the static and templates directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "static")
templates_dir = os.path.join(BASE_DIR, "templates")

app = FastAPI()

# Add these settings for the FastAPI app
app.root_path = ""
app.root_path_in_servers = True

# Define middleware classes first
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = "upgrade-insecure-requests"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

class BaseURLRedirectMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path == "/":
            if "spaces" in request.headers.get("host", ""):
                return RedirectResponse(url=request.url.path)
        return await call_next(request)

# Then add the middlewares in order
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(BaseURLRedirectMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://huggingface.co",
        "https://*.hf.space",
        "*"  # For development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Update the static files configuration
app.mount("/static", 
    StaticFiles(
        directory=static_dir, 
        html=True,
        check_dir=False
    ), 
    name="static"
)

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
async def root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "base_url": request.base_url,
        "is_hf_space": "spaces" in request.headers.get("host", "")
    })

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
                frame_data_url = data['frame']
                detection_enabled = data.get('detection', False)
                ocr_enabled = data.get('ocr', False)
                barcode_enabled = data.get('barcode', False)

                if not frame_data_url:
                    continue

                # Decode base64 data
                header, encoded = frame_data_url.split(',', 1)
                frame_data = base64.b64decode(encoded)
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
                    try:
                        ocr_results = ocr.ocr(img)
                        last_ocr_time = time.time()
                        
                        texts = []
                        if ocr_results and len(ocr_results) > 0 and ocr_results[0] is not None:
                            for line in ocr_results[0]:  # PaddleOCR returns a list of pages
                                try:
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
                                except Exception as line_error:
                                    print(f"Error processing OCR line: {str(line_error)}")
                                    continue

                        # Always send a response, even if no text is found
                        await websocket.send_json({
                            "type": "ocr",
                            "texts": texts if texts else ["No text detected"]
                        })
                    except Exception as ocr_error:
                        print(f"OCR processing error: {str(ocr_error)}")
                        await websocket.send_json({
                            "type": "ocr",
                            "texts": ["Error processing OCR"]
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
                print("WebSocket disconnected normally")
                break
            except Exception as e:
                print(f"WebSocket error: {str(e)}")
                # Send an error message to the client
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": "An error occurred processing the frame"
                    })
                except:
                    break  # Break if we can't send the error message
                continue

    except Exception as e:
        print(f"WebSocket connection error: {str(e)}")
    finally:
        try:
            await websocket.close()
        except:
            pass

@app.post("/analyze_snapshot")
async def analyze_snapshot(request: Request):
    try:
        data = await request.json()
        analysis_type = data.get('analysis_type', 'claude')  # Default to claude for backward compatibility
        
        if analysis_type == 'claude':
            image_data = np.frombuffer(bytes(data['frame']), np.uint8)
            img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            
            ocr_texts = data.get('ocr_texts', [])
            barcode_texts = data.get('barcode_texts', [])
            
            analysis = await ai_analyzer.analyze_snapshot_claude(img, ocr_texts, barcode_texts)
            return {"analysis": {"claude": analysis}}
            
        elif analysis_type == 'tavily':
            tavily_results = await ai_analyzer.analyze_snapshot_tavily()
            return {"analysis": {"tavily": tavily_results}}
            
    except Exception as e:
        return {"error": str(e)}

@app.post("/analyze_context")
async def analyze_context(request: Request):
    try:
        data = await request.json()
        question = data.get('question')
        
        # Get the context from the URLs
        context = await ai_analyzer.analyze_context(question)
        return {"summary": context}
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/static/styles.css")
async def serve_styles():
    return FileResponse(
        os.path.join(static_dir, "styles.css"),
        media_type="text/css",
        headers={
            "Content-Security-Policy": "upgrade-insecure-requests",
            "Cache-Control": "public, max-age=3600"
        }
    )
