import os
import uuid
import cv2
import numpy as np
from datetime import datetime


def ensure_dir(path):
    """
    Ensure directory exists, create if it doesn't

    """
    os.makedirs(path, exist_ok=True)
    return path


def timestamp_iso():
    """
    Get current UTC timestamp in ISO format
    """
    return datetime.utcnow().isoformat()


def save_cropped_face(img, prefix, logs_folder, save_cropped=True):
    """
    Save cropped face image to organized directory structure

    """
    if not save_cropped:
        return None
        
    # Create date-based directory structure: logs/{prefix}/{YYYY-MM-DD}/
    date_str = datetime.now().strftime("%Y-%m-%d")
    folder = os.path.join(logs_folder, prefix, date_str)
    ensure_dir(folder)
    
    # Generate unique filename with timestamp and UUID
    fname = f"{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
    path = os.path.join(folder, fname)
    
    # Save image as JPEG
    cv2.imwrite(path, img)
    return path


def cosine_similarity(a, b):
    """
    Calculate cosine similarity between two vectors

    """
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0  # Handle zero vectors
    
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))