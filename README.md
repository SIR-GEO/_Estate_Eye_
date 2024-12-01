# Estate Eyes 🏠

Estate Eyes is a state-of-the-art real-time computer vision application that integrates object detection, OCR (Optical Character Recognition), and barcode/QR code scanning with AI-driven analysis.

## Features 🌟

- **Real-Time Object Detection** powered by YOLOv8
- **Text Recognition** using PaddleOCR
- **Barcode & QR Code Scanning**
- **AI-Powered Analysis** with Claude 3 and Tavily
- **Context-Aware Search** functionality
- **Responsive Web Interface**
- **GPU Acceleration** for enhanced performance

## System Requirements 🖥️

- NVIDIA GPU with CUDA support (optional, but recommended)
- Python 3.11 or higher
- Docker for containerised deployment
- Minimum 4GB RAM
- Webcam or camera device

## Environment Variables 🔑

Create a `.env` file in the root directory with the following keys:
- `ANTHROPIC_API_KEY=your_claude_api_key`
- `TAVILY_API_KEY=your_tavily_api_key`

## Installation 🚀

### Using Docker

1. Build the Docker image:
   ```bash
   docker build -t estate-eyes .
   ```

2. Run the container:
   ```bash
   docker run -p 7860:7860 --gpus all estate-eyes
   ```

### Manual Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate # Linux/Mac
   venv\Scripts\activate # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 7860
   ```

## Architecture 🏗️

### Frontend
- HTML5 WebSocket-based real-time video streaming
- Responsive design with mobile support
- Real-time UI updates for detections and analysis

### Backend
- FastAPI for high-performance asynchronous operations
- WebSocket communication for real-time video processing
- Multi-threaded processing for concurrent analysis

### AI Components
- YOLOv8 for object detection
- PaddleOCR for text recognition
- Claude 3 for image and context analysis
- Tavily for web search integration

## Usage Guide 📖

1. **Camera Controls**
   - Toggle object detection
   - Toggle OCR
   - Toggle barcode/QR scanning
   - Switch between available cameras

2. **AI Analysis**
   - Click "LLM Search" to analyse the current frame
   - View AI analysis and related search results
   - Ask questions about the analysed content

3. **Search Context**
   - Enter questions in the text area
   - Receive AI-powered answers based on search results
   - View comprehensive summaries with emojis

## API Endpoints 🛣️

- `GET /` - Main application interface
- `WS /ws` - WebSocket endpoint for real-time video
- `POST /analyze_snapshot` - AI analysis of the current frame
- `POST /analyze_context` - Context-based question answering
- `GET /health` - Health check endpoint

## Performance Optimisation 🚄

- GPU acceleration for YOLO and PaddleOCR
- Efficient frame processing with OpenCV
- Asynchronous operations for improved responsiveness
- Memory-efficient PDF and web content processing

## Security Features 🔒

- CORS middleware with configurable origins
- Trusted host middleware
- Security headers middleware
- Environment variable protection

## Contributing 🤝

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Licence 📄

[Specify your licence information here]

## Acknowledgements 👏

- YOLOv8 by Ultralytics
- PaddleOCR by PaddlePaddle
- Claude 3 by Anthropic
- Tavily Search API