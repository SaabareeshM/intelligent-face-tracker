import os
import sys
import argparse
import logging
import cv2
from config import load_or_create_config
from database import DatabaseManager
from face_detector import FaceDetector
from tracker import FaceTracker
from visualizer import Visualizer
from utils import timestamp_iso, save_cropped_face, cosine_similarity

# Load configuration and ensure log directory exists
config = load_or_create_config()
os.makedirs("logs", exist_ok=True)

# Global progress tracking for Streamlit integration
current_progress = {"current_frame": 0, "total_frames": 0, "progress": 0.0}


class Utils:
    """Utility class for tracker compatibility"""
    
    @staticmethod
    def timestamp_iso():
        return timestamp_iso()
    
    @staticmethod
    def save_cropped_face(img, prefix, logs_folder, save_cropped):
        return save_cropped_face(img, prefix, logs_folder, save_cropped)


def setup_logging():
    """Configure logging to file and console"""
    logging.basicConfig(
        filename="events.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("face_pipeline")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(console_handler)
    return logger


def progress_callback(current_frame, total_frames):
    """Update progress tracking for external applications"""
    global current_progress
    if total_frames > 0:
        progress = (current_frame / total_frames) * 100
        current_progress["current_frame"] = current_frame
        current_progress["total_frames"] = total_frames
        current_progress["progress"] = progress
        if current_frame % 100 == 0:
            logging.getLogger("face_pipeline").info(f"Progress: {current_frame}/{total_frames} ({progress:.1f}%)")


def get_current_progress():
    """Get current progress state"""
    global current_progress
    return current_progress.copy()


def reset_progress():
    """Reset progress tracking to initial state"""
    global current_progress
    current_progress = {"current_frame": 0, "total_frames": 0, "progress": 0.0}


def process_video(source, max_frames=None, output_path=None, progress_callback=None):
    """
    Main video processing pipeline for face detection and tracking

    """
    logger = setup_logging()
    reset_progress()
    
    # Initialize pipeline components
    db_manager = DatabaseManager(config)
    face_detector = FaceDetector(config)
    utils = Utils()
    tracker = FaceTracker(config, db_manager, utils)
    visualizer = Visualizer(config, db_manager)
    
    # Open video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open video source: {source}")
        return None

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Handle camera streams with unknown frame count
    if total_frames == 0 and max_frames:
        total_frames = max_frames
    elif total_frames == 0:
        total_frames = 1000  # Default estimate for camera streams
    
    # Initialize video writer for output
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Load configuration parameters
    frame_num = 0
    detect_skip = max(1, int(config.get("detection_skip_frames", 5)))
    conf_thresh = float(config.get("detection_conf_threshold", 0.6))
    sim_thresh = float(config.get("embedding_similarity_threshold", 0.5))
    exit_thresh = int(config.get("exit_frame_threshold", 30))
    visualize = bool(config.get("visualize", True))

    # Load known face embeddings from database
    known_embeddings = db_manager.get_all_face_data()
    logger.info(f"Loaded {len(known_embeddings)} face vectors")

    # Initialize results tracking
    results = {
        'total_frames': total_frames,
        'processed_frames': 0,
        'unique_people': 0,
        'total_detections': 0,
        'output_path': output_path
    }

    # Main processing loop
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("End of stream")
            break
        frame_num += 1

        # Report progress to external applications
        if progress_callback and frame_num % 10 == 0:
            progress_callback(frame_num, total_frames)

        # Detect faces in current frame
        detections = face_detector.detect_faces(frame)
        results['total_detections'] += len(detections)
        tracker.update_trackers(frame, frame_num, detections)

        # Process recognition on detection frames or when no people are tracked
        if frame_num % detect_skip == 0 or not tracker.tracked_people:
            for x, y, w, h, conf in detections:
                # Skip low confidence detections
                if conf < conf_thresh:
                    continue
                    
                # Extract face region with padding
                pad = int(0.1 * max(w, h))
                xa, ya = max(0, x - pad), max(0, y - pad)
                xb, yb = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
                crop = frame[ya:yb, xa:xb]
                if crop.size == 0:
                    continue

                # Extract face embedding
                emb = face_detector.extract_embedding(crop)
                if emb is None:
                    continue

                # First try position-based matching with currently tracked people
                current_match = None
                for person_id, pdata in tracker.tracked_people.items():
                    if pdata.get("bbox"):
                        tracked_x, tracked_y, tracked_w, tracked_h = pdata["bbox"]
                        tracked_center_x = tracked_x + tracked_w / 2
                        tracked_center_y = tracked_y + tracked_h / 2
                        det_center_x = x + w / 2
                        det_center_y = y + h / 2
                        
                        # Calculate distance between detection and tracked face centers
                        distance = ((tracked_center_x - det_center_x) ** 2 + 
                                   (tracked_center_y - det_center_y) ** 2) ** 0.5
                        
                        # Match if centers are within 50 pixels
                        if distance < 50:
                            current_match = person_id
                            break

                # Assign ID based on matching strategy
                if current_match:
                    # Position match with existing track
                    assigned_id = current_match
                    is_new_person = False
                    logger.debug(f"Position match: {assigned_id}")
                else:
                    # No position match, try embedding-based recognition
                    best_id, best_sim, second_best_sim = db_manager.find_best_match(emb, known_embeddings)
                    
                    timestamp_now = timestamp_iso()
                    is_new_person = False

                    if best_sim >= sim_thresh and best_id:
                        # Embedding match with known person
                        assigned_id = best_id
                        logger.info(f"Embedding match: {assigned_id} (sim={best_sim:.3f})")
                    else:
                        # New person - register in database
                        assigned_id = db_manager.get_next_person_id()
                        is_new_person = True
                        db_manager.register_person(assigned_id, emb, timestamp_now)
                        known_embeddings.append((assigned_id, emb))
                        logger.info(f"New person: {assigned_id} (sim={best_sim:.3f})")

                # Log entry if this is a new presence in current tracking session
                if assigned_id not in tracker.tracked_people:
                    cropped_path = save_cropped_face(
                        crop, "entries", 
                        config["logs_folder"], 
                        config.get("save_cropped", True)
                    )
                    db_manager.save_visit_record(assigned_id, "entry", timestamp_now, cropped_path)
                    logger.info(f"Entry saved for {assigned_id}")

                # Store embedding if it's diverse enough from existing ones
                if not is_new_person:
                    if db_manager.should_store_embedding(assigned_id, emb):
                        try:
                            db_manager.db.face_data.insert_one({
                                "person_id": assigned_id,
                                "face_vector": emb.tolist(),
                                "created_time": timestamp_now
                            })
                            known_embeddings.append((assigned_id, emb))
                        except Exception as e:
                            logger.error(f"Storage error for {assigned_id}: {e}")

                # Update person's last seen timestamp and register for tracking
                db_manager.update_last_seen(assigned_id, timestamp_now)
                tracker.register_face(assigned_id, (x, y, w, h), crop, conf, frame, timestamp_now, frame_num)

        # Handle person exits based on frame threshold
        tracker.handle_exits(frame_num, exit_thresh)

        # Create visualization frame with annotations
        vis_frame = visualizer.draw_on_frame(frame, tracker.tracked_people, frame_num)

        # Display visualization if enabled
        if visualize:
            should_quit = visualizer.visualize(vis_frame, tracker.tracked_people, frame_num)
            if should_quit:
                break

        # Write frame to output video
        if out:
            out.write(vis_frame)

        # Stop if max frames reached
        if max_frames and frame_num >= max_frames:
            logger.info(f"Reached max_frames={max_frames}")
            break

    # Final cleanup - mark all remaining tracked people as exited
    tracker.handle_exits(frame_num, 0)
    
    # Update final results
    results['processed_frames'] = frame_num
    results['unique_people'] = db_manager.get_unique_visitor_count()

    # Release resources
    cap.release()
    if out:
        out.release()
    
    logger.info("Processing finished")
    return results


def process_video_with_progress(source, max_frames=None, output_path=None):
    """Wrapper function that includes progress callback for external applications"""
    return process_video(source, max_frames, output_path, progress_callback)


def main():
    """Command-line interface for the face detection pipeline"""
    parser = argparse.ArgumentParser(description="Face Detection Pipeline")
    parser.add_argument("--source", type=str, default=str(config.get("camera_source", 0)),
                        help="Video source (file path or camera index)")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Enable debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Convert source to integer if it's a camera index
    source = args.source
    if source.isdigit():
        source = int(source)
    
    print(f"Starting pipeline on: {source}")
    results = process_video_with_progress(source, max_frames=args.max_frames, output_path=args.output)
    
    # Display processing summary
    if results:
        print(f"Processing completed:")
        print(f"- Frames: {results['processed_frames']}")
        print(f"- Unique people: {results['unique_people']}")
        print(f"- Detections: {results['total_detections']}")


if __name__ == "__main__":
    main()