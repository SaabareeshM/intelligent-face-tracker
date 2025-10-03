import logging

logger = logging.getLogger("face_pipeline")


class FaceTracker:
    """Face tracking system using position-based matching and temporal consistency"""
    
    def __init__(self, config, db_manager, utils):
        self.config = config
        self.db_manager = db_manager
        self.utils = utils
        self.tracked_people = {}  # Active tracking sessions: {person_id: tracking_data}
    
    def update_trackers(self, frame, frame_num, current_detections=None):
        """
        Update tracking state for current frame

        """
        # Reset bounding boxes for all tracked people
        for person_id, pdata in self.tracked_people.items():
            pdata["bbox"] = None
        
        if current_detections:
            # Assign current detections to existing tracks
            self._assign_detections_to_tracks(current_detections, frame_num)
        else:
            # No detections - update timestamps for active tracks
            for person_id, pdata in self.tracked_people.items():
                if pdata.get("bbox"):
                    pdata["last_seen_frame"] = frame_num
                    self.db_manager.update_last_seen(person_id, self.utils.timestamp_iso())
    
    def _assign_detections_to_tracks(self, detections, frame_num):
        """
        Match current detections to existing tracks using position similarity

        """
        assigned_detections = set()  # Track which detections have been assigned
        
        for person_id, pdata in self.tracked_people.items():
            last_bbox = pdata.get("last_known_bbox")
            if not last_bbox:
                continue  # Skip if no previous position data
                
            best_match_idx = -1
            best_distance = float('inf')
            
            # Find best matching detection for this track
            for i, (x, y, w, h, conf) in enumerate(detections):
                if i in assigned_detections:
                    continue  # Skip already assigned detections
                
                # Calculate center points of last known position and current detection
                last_center_x = last_bbox[0] + last_bbox[2] / 2
                last_center_y = last_bbox[1] + last_bbox[3] / 2
                det_center_x = x + w / 2
                det_center_y = y + h / 2
                
                # Calculate Euclidean distance between centers
                distance = ((last_center_x - det_center_x) ** 2 + 
                           (last_center_y - det_center_y) ** 2) ** 0.5
                
                # Match if within 100 pixels and closest so far
                if distance < 100 and distance < best_distance:
                    best_distance = distance
                    best_match_idx = i
            
            # Assign best matching detection to this track
            if best_match_idx != -1:
                x, y, w, h, conf = detections[best_match_idx]
                pdata["bbox"] = (x, y, w, h)  # Current position
                pdata["last_known_bbox"] = (x, y, w, h)  # Update reference position
                pdata["last_seen_frame"] = frame_num
                pdata["conf"] = conf
                assigned_detections.add(best_match_idx)
                logger.debug(f"Matched detection {best_match_idx} to {person_id}")
    
    def handle_exits(self, current_frame_num, exit_thresh):
        """
        Handle person exits based on frame threshold

        """
        to_remove = []
        for person_id, pdata in self.tracked_people.items():
            last_seen = pdata.get("last_seen_frame", 0)
            
            # Check if person hasn't been seen for threshold frames
            if current_frame_num - last_seen > exit_thresh:
                logger.info(f"Person {person_id} EXIT at frame {current_frame_num}")
                
                # Save exit record with cropped face if available
                crop = pdata.get("last_crop")
                img_path = self.utils.save_cropped_face(
                    crop, "exits", 
                    self.config.get("logs_folder", "logs"),
                    self.config.get("save_cropped", True)
                ) if crop is not None else None
                
                timestamp_now = self.utils.timestamp_iso()
                self.db_manager.save_visit_record(person_id, "exit", timestamp_now, img_path)
                logger.info(f"Exit saved for {person_id}")
                to_remove.append(person_id)
        
        # Remove exited people from tracking
        for person_id in to_remove:
            self.tracked_people.pop(person_id, None)
    
    def register_face(self, person_id, bbox, crop, conf, frame, timestamp_now, frame_num):
        """
        Register a new person for tracking or update existing track

        """
        self.tracked_people[person_id] = {
            "last_seen_frame": frame_num,    # Last frame where person was detected
            "bbox": bbox,                    # Current bounding box
            "last_known_bbox": bbox,         # Reference position for matching
            "last_crop": crop,               # Last cropped face image
            "conf": conf,                    # Detection confidence
            "last_seen_time": timestamp_now, # Last seen timestamp
        }
        logger.debug(f"Registered {person_id} for tracking")
    
    def get_tracked_people(self):
        """Get currently tracked people data"""
        return self.tracked_people