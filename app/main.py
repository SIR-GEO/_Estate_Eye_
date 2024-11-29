from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
import time
import asyncio
from ultralytics import YOLO
import torch

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize FastAPI app
app = FastAPI()

# HTML for client-side rendering
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>YOLO Real-Time Detection</title>
        <style>
            body {
                background-color: #00274d;
                color: white;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 100vh;
                margin: 0;
                font-family: Arial, sans-serif;
            }
            .video-container {
                display: flex;
                flex-direction: row;
                justify-content: center;
                align-items: center;
                gap: 20px;
            }
            .video-wrapper {
                text-align: center;
                position: relative;
            }
            video, canvas {
                border: 2px solid #ffffff;
                border-radius: 8px;
            }
            .fps-counter {
                position: absolute;
                top: 10px;
                left: 10px;
                background: rgba(0, 0, 0, 0.5);
                color: white;
                padding: 5px;
                border-radius: 5px;
                font-family: Arial, sans-serif;
                font-size: 14px;
            }
            h1 {
                margin-bottom: 20px;
            }
        </style>
    </head>
    <body>
        <h1>YOLO Real-Time Object Detection</h1>
        <div class="video-container">
            <div class="video-wrapper">
                <label for="video">Original Webcam Feed</label><br>
                <video id="video" autoplay muted playsinline style="width: 640px; height: 480px;"></video>
                <div id="fps-counter" class="fps-counter">FPS: --</div>
            </div>
            <div class="video-wrapper">
                <label for="canvas">Processed Detection Feed</label><br>
                <canvas id="canvas" style="width: 640px; height: 480px;"></canvas>
            </div>
        </div>
        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const context = canvas.getContext('2d');
            const fpsCounter = document.getElementById('fps-counter');
            let websocket;

            async function startVideo() {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = stream;
                video.play();

                const track = stream.getVideoTracks()[0];
                const imageCapture = new ImageCapture(track);

                // Establish WebSocket connection
                connectWebSocket();

                // Continuously send frames to the server
                let lastFrameTime = performance.now();
                setInterval(async () => {
                    const now = performance.now();
                    const fps = (1000 / (now - lastFrameTime)).toFixed(1);
                    lastFrameTime = now;
                    fpsCounter.textContent = `FPS: ${fps}`;

                    const blob = await imageCapture.grabFrame();
                    const canvasBlob = await createImageBitmap(blob);
                    canvas.width = canvasBlob.width;
                    canvas.height = canvasBlob.height;
                    context.drawImage(canvasBlob, 0, 0);

                    canvas.toBlob((blob) => {
                        if (websocket.readyState === WebSocket.OPEN) {
                            websocket.send(blob);
                        }
                    }, "image/jpeg");
                }, 100);
            }

            function connectWebSocket() {
                websocket = new WebSocket("ws://localhost:8000/ws");

                websocket.onmessage = (event) => {
                    const img = new Image();
                    img.src = URL.createObjectURL(event.data);
                    img.onload = () => {
                        context.drawImage(img, 0, 0);
                    };
                };

                websocket.onclose = () => {
                    console.warn("WebSocket closed. Reconnecting...");
                    setTimeout(connectWebSocket, 1000);
                };
            }

            startVideo();
        </script>
    </body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()

        while True:
            try:
                # Receive and decode frame
                frame = await websocket.receive_bytes()
                nparr = np.frombuffer(frame, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Perform object detection
                results = model.predict(img, conf=0.5, device=device, stream=True)

                # Draw bounding boxes
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = model.names[int(box.cls[0])]
                        score = box.conf[0]

                        # Draw rectangle and label
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
                await websocket.send_bytes(buffer.tobytes())

            except Exception as e:
                print(f"Frame processing error: {e}")
                continue

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
