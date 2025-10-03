import os
import json

DEFAULT_CONFIG = {
    # Face detection settings
    "detection_skip_frames": 5,           # Process every Nth frame for performance
    "detection_conf_threshold": 0.6,      # Minimum confidence for face detection
    "embedding_similarity_threshold": 0.5, # Threshold for face recognition matching
    "embedding_diversity_threshold": 0.8,  # Store only diverse face embeddings
    
    # Tracking and logging settings
    "exit_frame_threshold": 30,           # Frames before marking person as exited
    "save_cropped": True,                 # Save cropped face images
    "logs_folder": "logs",                # Directory for log files
    
    # Database configuration
    "mongodb_uri": "mongodb://localhost:27017/",
    "database_name": "face_tracker",
    
    # Model configuration
    "model_yolo": "yolov8n-face.pt",      # YOLO model for face detection
    "det_size": 640,                      # Detection input size
    
    # Display and input settings
    "visualize": True,                    # Show real-time visualization
    "camera_source": 0                    # Default camera index
}

CONFIG_PATH = "config.json"


def load_or_create_config(path=CONFIG_PATH):
    if os.path.exists(path):
        # Load existing configuration
        with open(path, "r") as f:
            cfg = json.load(f)
        
        # Ensure all default parameters are present
        for k, v in DEFAULT_CONFIG.items():
            if k not in cfg:
                cfg[k] = v
        return cfg
    else:
        # Create default configuration file
        with open(path, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        print(f"Created default config at {path}")
        return DEFAULT_CONFIG.copy()