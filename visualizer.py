import cv2
from datetime import datetime


class Visualizer:
    """Visualization system for face tracking results and statistics"""
    
    def __init__(self, config, db_manager):
        self.config = config
        self.db_manager = db_manager
        
        # Color scheme for visual elements (BGR format)
        self.colors = {
            "bbox_color": (255, 0, 0),      # Blue - Face bounding boxes
            "text_color": (0, 0, 255),      # Red - Primary text
            "text_color2": (255, 255, 255), # White - Secondary text  
            "text_bg_color": (255, 0, 0)    # Blue - Text backgrounds
        }
    
    def set_colors(self, bbox_color=None, text_color=None, text_color2=None, text_bg_color=None):
        """Update visualization colors dynamically"""
        if bbox_color:
            self.colors["bbox_color"] = bbox_color
        if text_color:
            self.colors["text_color"] = text_color
        if text_color2:
            self.colors["text_color2"] = text_color2
        if text_bg_color:
            self.colors["text_bg_color"] = text_bg_color
    
    def draw_on_frame(self, frame, tracked_people, frame_num):
        """
        Draw face tracking annotations on frame without displaying

        """
        vis = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 3
        
        # Extract colors for drawing
        bbox_color = self.colors["bbox_color"]
        text_color = self.colors["text_color"]
        text_color2 = self.colors["text_color2"]
        text_bg_color = self.colors["text_bg_color"]

        frame_h, frame_w = vis.shape[:2]

        # Top-left: Unique visitor count from database
        visitors_text = f"Visitors: {self.db_manager.get_unique_visitor_count()}"
        cv2.putText(vis, visitors_text, (10, 30), font, font_scale, text_color, font_thickness)

        # Top-right: Frame number (right-aligned)
        frame_text = f"Frame: {frame_num}"
        (text_w, _), _ = cv2.getTextSize(frame_text, font, font_scale, font_thickness)
        cv2.putText(vis, frame_text, (frame_w - text_w - 10, 30), font, font_scale, text_color, font_thickness)

        # Bottom-left: Current timestamp
        timestamp_text = f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        cv2.putText(vis, timestamp_text, (10, frame_h - 10), font, font_scale, text_color, font_thickness)

        # Draw bounding boxes and labels for each tracked person
        for person_id, pdata in tracked_people.items():
            bb = pdata.get("bbox")
            if not bb:
                continue  # Skip if no current bounding box
                
            x, y, w, h = bb

            # Draw face bounding box
            cv2.rectangle(vis, (x, y), (x+w, y+h), bbox_color, 3)

            # Draw person ID label above bounding box
            label = f"{person_id}"
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Draw background for text readability
            cv2.rectangle(vis, (x, y - text_h - 10), (x + text_w, y), text_bg_color, -1)
            
            # Draw person ID text
            cv2.putText(vis, label, (x, y - 5), font, font_scale, text_color, font_thickness)

        # Statistics overlay - count of currently visible people
        inframe_people = len([1 for p in tracked_people.values() if p.get("bbox") is not None])
        inframe_text = f"In-frame: {inframe_people}"
        cv2.putText(vis, inframe_text, (10, 60), font, font_scale, text_color, font_thickness)

        # Entry and exit statistics from database
        entry_count = self.db_manager.get_visit_count("entry")
        exit_count = self.db_manager.get_visit_count("exit")

        cv2.putText(vis, f"Entries: {entry_count}", (10, 90), font, font_scale, text_color2, font_thickness)
        cv2.putText(vis, f"Exits: {exit_count}", (10, 120), font, font_scale, text_color2, font_thickness)

        return vis
    
    def visualize(self, frame, tracked_people, frame_num):
        """
        Display annotated frame in OpenCV window

        """
        if not self.config.get("visualize", True):
            return False  # Visualization disabled in config
        
        # Create annotated frame
        vis = self.draw_on_frame(frame, tracked_people, frame_num)

        # Display in fullscreen window
        cv2.namedWindow("Face Pipeline", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Face Pipeline", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Face Pipeline", vis)

        # Check for quit key (q)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return True
        return False