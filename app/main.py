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

# Get the absolute path to the static and templates directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "static")
templates_dir = os.path.join(BASE_DIR, "templates")

app = FastAPI()

# Mount static files with absolute path
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# Initialize model
model, device = initialize_model()

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        connection_active = True

        while connection_active:
            try:
                # Receive and decode frame
                data = await websocket.receive_json()
                frame_data = bytes(data['frame'])
                detection_enabled = data['detection']

                nparr = np.frombuffer(frame_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if detection_enabled:
                    # Perform object detection
                    results = model.predict(img, conf=0.5, device=device, stream=True)

                    # Draw bounding boxes
                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            label = model.names[int(box.cls[0])]
                            score = box.conf[0]

                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(
                                img,
                                f"{label} ({score:.2f})",
                                (x1, max(y1 - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2,
                            )

                # Send processed frame
                _, buffer = cv2.imencode(".jpg", img)
                try:
                    await websocket.send_bytes(buffer.tobytes())
                except WebSocketDisconnect:
                    connection_active = False

            except WebSocketDisconnect:
                connection_active = False
                print("WebSocket disconnected - inner loop")
                break
            except Exception as e:
                print(f"Frame processing error: {str(e)}")
                if "disconnect" in str(e).lower():
                    connection_active = False
                    break
                continue

    except WebSocketDisconnect:
        print("WebSocket disconnected - outer loop")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
    finally:
        print("WebSocket connection closed")
