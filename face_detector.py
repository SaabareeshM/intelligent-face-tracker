import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import logging

logger = logging.getLogger("face_pipeline")


class FaceDetector:
    """Face detection and embedding extraction using YOLO and InsightFace"""
    
    def __init__(self, config):
        self.config = config
        self.yolo = self._init_yolo()
        self.face_app = self._init_insightface()
    
    def _init_yolo(self):
        """Initialize YOLO face detection model"""
        model_name = self.config.get("model_yolo", "yolov8n-face.pt")
        yolo = YOLO(model_name)
        logger.info("YOLO initialized")
        return yolo
    
    def _init_insightface(self):
        """Initialize InsightFace for face recognition and embedding extraction"""
        face_app = FaceAnalysis(name='buffalo_l')
        det_size = self.config.get("det_size", 640)
        
        # Try GPU first, fallback to CPU if unavailable
        try:
            face_app.prepare(ctx_id=0, det_size=(det_size, det_size))
        except Exception:
            logger.warning("GPU not available, using CPU")
            face_app.prepare(ctx_id=-1, det_size=(det_size, det_size))
        
        logger.info("InsightFace initialized")
        return face_app
    
    def detect_faces(self, frame):
        """
        Detect faces in frame using YOLO model
        """
        results = self.yolo.predict(
            frame, 
            imgsz=self.config.get("det_size", 640),
            conf=self.config.get("detection_conf_threshold", 0.6),
            verbose=False
        )
        
        detections = []
        if results:
            r = results[0]
            if hasattr(r, "boxes"):
                for b in r.boxes:
                    # Filter for face class (class 0)
                    if b.cls != 0:
                        continue
                    
                    # Extract bounding box coordinates and confidence
                    xyxy = b.xyxy.cpu().numpy().astype(int).flatten()
                    conf = float(b.conf.cpu().numpy())
                    
                    # Apply confidence threshold
                    if conf < self.config.get("detection_conf_threshold", 0.6):
                        continue
                    
                    # Convert to (x, y, w, h) format
                    x1, y1, x2, y2 = xyxy[:4]
                    w, h = x2 - x1, y2 - y1
                    x1, y1 = max(0, x1), max(0, y1)  # Ensure positive coordinates
                    
                    detections.append((x1, y1, w, h, conf))
        
        return detections
    
    def extract_embedding(self, crop):
        """
        Extract face embedding from cropped face image
        """
        faces = self.face_app.get(crop) or []
        
        if faces:
            # Select face with highest detection score
            best_face = sorted(faces, key=lambda f: f.det_score, reverse=True)[0]
            emb = getattr(best_face, "embedding", None)
            
            if emb is not None:
                return np.asarray(emb, dtype=np.float32)
        
        return None