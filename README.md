# 🧠 Intelligent Face Tracker with Auto-Registration and Visitor Counting

An **AI-driven real-time face detection, recognition, and tracking system** that automatically registers new faces, tracks them across video frames, and maintains accurate unique visitor counts.

---

## 🚀 Features
- **Real-time Face Detection**: YOLOv8-based face detection  
- **Face Recognition**: InsightFace for facial embeddings and recognition  
- **Multi-face Tracking**: Custom tracking logic for continuous face following  
- **Auto-registration**: Automatic registration of new faces  
- **Visitor Counting**: Accurate unique visitor counting  
- **Comprehensive Logging**: Entry/exit logs with timestamps and images  
- **Database Storage**: MongoDB for embeddings and metadata  
- **RTSP Support**: Live camera stream processing capability  
- **Web Interface**: Streamlit-based dashboard for analytics  

---

## 🛠️ Setup Instructions

### 1. Prerequisites
- Python **3.8 – 3.10**  
- Webcam or video file for testing  
- MongoDB (local or cloud, e.g., Atlas)  

### 2. Installation

# Clone the repository
git clone https://github.com/SaabareeshM/intelligent-face-tracker.git
cd intelligent-face-tracker

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Running the Project

# Run face tracker
python main.py

# Run web interface
streamlit run app.py

# Sample config.json
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


# 🏗️ Architecture Diagram

Video Input
    │
    ▼
YOLOv8 → Face Detection
    │
    ▼
InsightFace → Face Embeddings
    │
    ▼
Custom Tracker → Recognition & Matching
    │
    ├── MongoDB: Store embeddings, visit records
    ├── Logs: Entry/Exit events, cropped images
    └── Visitor Counter

# 🗄️ MongoDB Schema

# 1. people Collection

{
  "person_id": "unique_id",
  "first_seen": "timestamp",
  "last_seen": "timestamp",
  "visit_count": 5
}


# 2. face_data Collection

{
  "person_id": "unique_id",
  "face_vector": [0.123, 0.456, ...],
  "created_time": "timestamp"
}


# 3. visit_records Collection

{
  "person_id": "unique_id",
  "action": "entry/exit",
  "timestamp": "timestamp",
  "photo_path": "logs/entries/face_123.jpg"
}


# 4. counter Collection

{
  "name": "person_counter",
  "current_number": 45
}


# 🔄 Relationships

people ←→ face_data (one-to-many)

people ←→ visit_records (one-to-many)

counter is standalone for unique ID generation

# 🎥 Demo Video

👉 Loom/YouTube Link Here

# 📂 Project Structure

FACE_TRACKER/
│
├── logs/                # Logs folder (entries/exits)
├── output_videos/       # Processed videos
├── temp_videos/         # Temporary video files
├── videos/              # Input videos
│
├── app.py               # Streamlit/Flask entry point
├── config.json          # Config file
├── config.py            # Config handler
├── database.py          # DB operations
├── events.log           # Event logs
├── face_detector.py     # Face detection logic
├── main.py              # Main script
├── tracker.py           # Face tracking
├── utils.py             # Helper functions
├── visualizer.py        # Visualization utilities
├── yolov8n-face.pt      # YOLOv8 pretrained face model
├── requirements.txt     # Dependencies
└── README.md            # Documentation

# ⚡ Hackathon Note

This project is a part of a hackathon run by https://katomaran.com

