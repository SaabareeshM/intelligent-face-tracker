# 🧠 Intelligent Face Tracker with Auto-Registration and Visitor Counting

An **AI-driven real-time face detection, recognition, and tracking system** that automatically registers new faces, tracks them across video frames, and maintains accurate unique visitor counts.

---

## 🚀 Features

- **Real-time Face Detection**: YOLOv8-based face detection  
- **Face Recognition**: InsightFace for facial embeddings and recognition  
- **Multi-face Tracking**: Custom tracking (Face Embedding + Cosine Similarity)  
- **Auto-registration**: Automatic registration of new faces  
- **Visitor Counting**: Accurate unique visitor counting  
- **Comprehensive Logging**: Entry/exit logs with timestamps and images  
- **Database Storage**: MongoDB for embeddings and metadata  
- **RTSP Support**: Live camera stream processing capability  
- **Web Interface**: Streamlit-based dashboard for analytics  

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.10
- Webcam or video file for testing
- MongoDB local

### Installation Steps
```bash
# Clone the repository
git clone https://github.com/SaabareeshM/intelligent-face-tracker.git
cd intelligent-face-tracker

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Run face tracker (main application)
python main.py

# Run web dashboard
streamlit run app.py
```

---

## ⚙️ Configuration

Create `config.json` file in the project root:
```json
{
  "detection_skip_frames": 3,
  "detection_conf_threshold": 0.3,
  "embedding_similarity_threshold": 0.4,
  "embedding_diversity_threshold": 0.8,
  "exit_frame_threshold": 15,
  "save_cropped": true,
  "logs_folder": "logs",
  "mongodb_uri": "mongodb://localhost:27017",
  "database_name": "face_tracker",
  "model_yolo": "yolov8n-face.pt",
  "det_size": 512,
  "visualize": true,
  "camera_source": 0
}
```

---

## 🏗️ System Architecture

```
Video Input (RTSP/File/Camera)
         │
         ▼
   Face Detection (YOLOv8)
         │
         ▼
Face Recognition (InsightFace)
         │
         ▼
  Custom Tracking Logic
         │
    ┌────┴────┐
    ▼         ▼
 MongoDB   File System
 Storage     Logging
    │           │
    ▼           ▼
 Analytics  Face Images
 & Counting  (Entries/Exits)
```

---

## 🗄️ Database Schema

### Collections in MongoDB:

#### 1. **people** 
```json
{
  "person_id": "person1",
  "first_seen": "2024-01-01T10:30:00",
  "last_seen": "2024-01-01T10:35:00",
  "visit_count": 3
}
```

#### 2. **face_data** 
```json
{
  "person_id": "person1",
  "face_vector": [0.123, 0.456, 0.789, ...],
  "created_time": "2024-01-01T10:30:00"
}
```

#### 3. **visit_records**
```json
{
  "person_id": "person1",
  "action": "entry",
  "timestamp": "2024-01-01T10:30:00",
  "photo_path": "logs/entries/person1_20240101_103000.jpg"
}
```

#### 4. **counter**
```json
{
  "name": "person_counter",
  "current_number": 45
}
```

### 🔗 Data Relationships
- **people** ↔ **face_data** (one-to-many)
- **people** ↔ **visit_records** (one-to-many)
- **counter** (standalone for ID generation)

---

## 📂 Project Structure

```
FACE_TRACKER/
│
├── logs/                      # Auto-generated logs
│   ├── entries/               # Entry face images
│   └── exits/                 # Exit face images
│
├── output_videos/             # Processed output videos
├── temp_videos/               # Temporary video files  
├── videos/                    # Input video samples
│
├── app.py                     # Streamlit web interface
├── config.json                # Configuration file
├── config.py                  # Configuration handler
├── database.py                # MongoDB operations
├── events.log                 # System event logs
├── face_detector.py           # YOLO face detection
├── main.py                    # Main application entry
├── tracker.py                 # Custom face tracking
├── utils.py                   # Utility functions
├── visualizer.py              # Visualization utilities
├── yolov8n-face.pt            # YOLOv8 face detection model
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 📊 Constraints

- **Lighting**: Reasonably well-lit environments for optimal detection
- **Face Angles**: Works best with frontal or near-frontal face angles
- **Video Quality**: Minimum 480p resolution for reliable detection
- **Processing**: Real-time performance benefits from CUDA-capable GPU
- **Camera**: Relatively stable camera position for consistent tracking

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Face Detection | YOLOv8 |
| Face Recognition | InsightFace |
| Tracking | Face Embedding + Cosine Similarity |
| Database | MongoDB |
| Computer Vision | OpenCV |
| Web Interface | Streamlit |
| Programming | Python |

---

## 🎥 Demo Video

[![Watch Demo](https://img.shields.io/badge/🎥-Watch_Demo_Video-red)](YOUR_LOOM_OR_YOUTUBE_LINK_HERE)

---

> **This project is a part of a hackathon run by [https://katomaran.com](https://katomaran.com)**

---
