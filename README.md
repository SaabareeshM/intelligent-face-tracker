# 🧠 Intelligent Face Tracker with Auto-Registration and Visitor Counting

description: "An AI-driven real-time face detection, recognition, and tracking system that automatically registers new faces, tracks them across video frames, and maintains accurate unique visitor counts"

features:
  - "Real-time Face Detection: YOLOv8-based face detection"
  - "Face Recognition: InsightFace for facial embeddings and recognition"
  - "Multi-face Tracking: Custom tracking logic for continuous face following"
  - "Auto-registration: Automatic registration of new faces"
  - "Visitor Counting: Accurate unique visitor counting"
  - "Comprehensive Logging: Entry/exit logs with timestamps and images"
  - "Database Storage: MongoDB for embeddings and metadata"
  - "RTSP Support: Live camera stream processing capability"
  - "Web Interface: Streamlit-based dashboard for analytics"

setup:
  prerequisites:
    - "Python 3.8 – 3.10"
    - "Webcam or video file for testing"
    - "MongoDB (local or cloud, e.g., Atlas)"
  
  installation: |
    # Clone the repository
    git clone https://github.com/SaabareeshM/intelligent-face-tracker.git
    cd intelligent-face-tracker

    # Create virtual environment
    python -m venv venv
    venv\Scripts\activate      # On Windows
    source venv/bin/activate   # On Linux/Mac

    # Install dependencies
    pip install -r requirements.txt

running:
  commands:
    - "python main.py"
    - "streamlit run app.py"

configuration:
  file: "config.json"
  sample: |
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

architecture: |
  Video Input (RTSP/File)
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
      ├── MongoDB Storage
      │   ├── people collection
      │   ├── face_data collection  
      │   ├── visit_records collection
      │   └── counter collection
      │
      ├── File System Logging
      │   ├── entries/ (cropped faces)
      │   └── exits/ (cropped faces)
      │
      └── Visitor Counting & Analytics

database:
  schema:
    people_collection:
      - "person_id: string (unique)"
      - "first_seen: timestamp"
      - "last_seen: timestamp"
      - "visit_count: integer"
    
    face_data_collection:
      - "person_id: string"
      - "face_vector: array[512]"
      - "created_time: timestamp"
    
    visit_records_collection:
      - "person_id: string"
      - "action: string (entry/exit)"
      - "timestamp: timestamp"
      - "photo_path: string"
    
    counter_collection:
      - "name: string"
      - "current_number: integer"

relationships:
  - "people ↔ face_data (one-to-many)"
  - "people ↔ visit_records (one-to-many)"
  - "counter (standalone for ID generation)"

demo_video: "YOUR_LOOM_OR_YOUTUBE_LINK_HERE"

project_structure: |
  FACE_TRACKER/
  │
  ├── logs/                       # Auto-generated logs
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

assumptions:
  - "Lighting Conditions: Reasonably well-lit environments for optimal detection"
  - "Face Angles: Works best with frontal or near-frontal face angles"
  - "Video Quality: Minimum 480p resolution for reliable detection"
  - "Processing: Real-time performance benefits from CUDA-capable GPU"
  - "Camera Stability: Relatively stable camera position for consistent tracking"

technologies:
  - "Face Detection: YOLOv8"
  - "Face Recognition: InsightFace"
  - "Tracking: Custom tracking algorithm"
  - "Database: MongoDB"
  - "Computer Vision: OpenCV"
  - "Web Interface: Streamlit"
  - "Programming: Python"

license: "MIT"

hackathon_note: "This project is a part of a hackathon run by https://katomaran.com"
