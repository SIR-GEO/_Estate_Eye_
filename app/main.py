from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
import torch

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1

# Load YOLO object detection pipeline
detector = pipeline(task="object-detection", model="hustvl/yolos-tiny", device=device)

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
            }
            video, canvas {
                border: 2px solid #ffffff;
                border-radius: 8px;
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
            const websocket = new WebSocket("ws://localhost:8000/ws");

            async function startVideo() {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = stream;
                video.play();

                const track = stream.getVideoTracks()[0];
                const imageCapture = new ImageCapture(track);

                // Continuously send frames to the server
                setInterval(async () => {
                    const blob = await imageCapture.grabFrame();
                    const canvasBlob = await createImageBitmap(blob);
                    canvas.width = canvasBlob.width;
                    canvas.height = canvasBlob.height;
                    context.drawImage(canvasBlob, 0, 0);
                    
                    canvas.toBlob((blob) => {
                        websocket.send(blob);
                    }, 'image/jpeg');
                }, 100);
            }

            websocket.onmessage = (event) => {
                const img = new Image();
                img.src = URL.createObjectURL(event.data);
                img.onload = () => {
                    context.drawImage(img, 0, 0);
                };
            };

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
    await websocket.accept()

    while True:
        try:
            # Receive the frame from the client
            frame = await websocket.receive_bytes()
            nparr = np.frombuffer(frame, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Convert OpenCV frame to RGB and PIL image
            frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Perform object detection
            results = detector(pil_image)

            # Draw bounding boxes on the frame
            for result in results:
                box = result["box"]
                label = result["label"]
                score = result["score"]

                x1, y1, x2, y2 = int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{label} ({score:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Encode the processed frame as JPEG and send it back
            _, buffer = cv2.imencode('.jpg', img)
            await websocket.send_bytes(buffer.tobytes())

        except Exception as e:
            print(f"Error: {e}")
            break
