import streamlit as st
import cv2
import os
from datetime import datetime
from main import process_video, get_current_progress, reset_progress
from config import load_or_create_config
from database import DatabaseManager
import json
import sys

# Streamlit page configuration
st.set_page_config(
    page_title="Face Tracker AI",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .success-box {
        background-color: #93daa4;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #83ebff;
        border: 1px solid #bee5eb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
    }
    .upload-section {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class StreamlitApp:
    """Main Streamlit application class for Face Tracker AI"""
    
    def __init__(self):
        self.config = load_or_create_config()
        self.db_manager = DatabaseManager(self.config)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories for file storage"""
        os.makedirs("output_videos", exist_ok=True)
        os.makedirs("temp_videos", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
    
    def render_sidebar(self):
        """Render the sidebar navigation and controls"""
        st.sidebar.title("🎯 Face Tracker AI")
        
        # App navigation
        app_mode = st.sidebar.selectbox(
            "Choose App Mode",
            ["🏠 Dashboard", "⚙️ Settings"]
        )
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Video Processing")
        
        # Initialize session state for processing control
        if 'processing' not in st.session_state:
            st.session_state.processing = False
        if 'processed_video_path' not in st.session_state:
            st.session_state.processed_video_path = None
        
        # Stop processing button when active
        if st.session_state.processing:
            if st.sidebar.button("🛑 Stop Processing", key="stop_processing"):
                st.session_state.processing = False
                st.rerun()
        
        st.sidebar.markdown("---")
        st.sidebar.info("Real-time face detection & tracking with MongoDB storage")
        
        return app_mode
    
    def render_dashboard(self):
        """Render the main dashboard with metrics and video processing"""
        st.markdown('<h1 class="main-header">👤 Face Tracker AI</h1>', unsafe_allow_html=True)
        
        self.render_metrics()
        self.render_video_processing()
        
        # Split layout for results and activity
        col1, col2 = st.columns([2, 1])
        with col1:
            self.render_processing_results()
        with col2:
            self.render_recent_activity()
    
    def render_metrics(self):
        """Display key performance metrics from database"""
        try:
            unique_visitors = self.db_manager.get_unique_visitor_count()
            entry_count = self.db_manager.get_visit_count("entry")
            exit_count = self.db_manager.get_visit_count("exit")
            current_in_frame = max(0, entry_count - exit_count)
            
            # Display metrics in cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">👥 Unique Visitors</div>
                    <div class="metric-value">{unique_visitors}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">📥 Total Entries</div>
                    <div class="metric-value">{entry_count}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">📤 Total Exits</div>
                    <div class="metric-value">{exit_count}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">🎯 In-Frame</div>
                    <div class="metric-value">{current_in_frame}</div>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error loading metrics: {e}")
    
    def render_video_processing(self):
        """Render video upload and camera processing sections"""
        st.markdown("---")
        st.markdown('<div class="sub-header">🎥 Video Processing</div>', unsafe_allow_html=True)
        
        # Tab interface for different input methods
        tab1, tab2 = st.tabs(["📁 Upload Video", "📷 Camera Input"])
        
        with tab1:
            self.render_upload_section()
        with tab2:
            self.render_camera_section()
    
    def render_upload_section(self):
        """Render video file upload interface"""
        st.markdown("""
        <div class="upload-section">
            <h3>Upload Video for Processing</h3>
            <p>Supported formats: MP4, AVI, MOV, MKV</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose video file", 
            type=['mp4', 'avi', 'mov', 'mkv'],
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.video(uploaded_file)
                st.info(f"File: {uploaded_file.name}")
            with col2:
                if st.button("🚀 Process Video", key="process_upload", type="primary"):
                    self.process_uploaded_video(uploaded_file)
    
    def render_camera_section(self):
        """Render live camera processing interface"""
        st.markdown("""
        <div class="info-box">
            <h4>📷 Camera Processing</h4>
            <p>Process live camera feed for real-time face tracking.</p>
        </div>
        """, unsafe_allow_html=True)
        
        camera_index = st.selectbox("Select Camera", options=[0, 1, 2], index=0)
        process_frames = st.slider("Number of frames to process", 50, 500, 100)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎬 Start Camera Processing", key="start_camera"):
                self.process_camera_feed(camera_index, process_frames)
        with col2:
            if st.session_state.processing:
                if st.button("🛑 Stop Camera", key="stop_camera"):
                    st.session_state.processing = False
                    st.rerun()
    
    def process_uploaded_video(self, uploaded_file):
        """Process uploaded video file through face detection pipeline"""
        try:
            reset_progress()
            st.session_state.processing = True
            st.session_state.processed_video_path = None
            
            # Save uploaded file to temporary location
            temp_dir = "temp_videos"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.name}")
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Define output path for processed video
            output_dir = "output_videos"
            os.makedirs(output_dir, exist_ok=True)
            output_filename = f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.name}"
            output_path = os.path.join(output_dir, output_filename)
            
            # Process video through main pipeline
            with st.spinner("Processing video..."):
                results = process_video(source=temp_path, output_path=output_path)
            
            # Cleanup temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if results and os.path.exists(output_path):
                st.session_state.processing = False
                st.session_state.processed_video_path = output_path
                st.session_state.processing_results = results
                st.success("Video processing completed!")
                st.rerun()
            else:
                st.session_state.processing = False
                st.error("Video processing failed")
                
        except Exception as e:
            st.session_state.processing = False
            st.error(f"Error: {str(e)}")
    
    def process_camera_feed(self, camera_index, max_frames):
        """Process live camera feed for face detection"""
        try:
            reset_progress()
            st.session_state.processing = True
            
            output_dir = "output_videos"
            os.makedirs(output_dir, exist_ok=True)
            output_filename = f"camera_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            output_path = os.path.join(output_dir, output_filename)
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process camera feed
            results = process_video(
                source=camera_index,
                max_frames=max_frames,
                output_path=output_path
            )
            
            progress_bar.progress(100)
            status_text.text("Processing completed!")
            
            if results and os.path.exists(output_path):
                st.session_state.processing = False
                st.session_state.processed_video_path = output_path
                st.session_state.processing_results = results
                st.success("Camera processing completed!")
                st.rerun()
            else:
                st.session_state.processing = False
                st.error("Camera processing failed")
                
        except Exception as e:
            st.session_state.processing = False
            st.error(f"Camera error: {str(e)}")
    
    def render_processing_results(self):
        """Display processing results and download options"""
        if hasattr(st.session_state, 'processed_video_path') and st.session_state.processed_video_path:
            st.markdown("---")
            st.markdown('<div class="sub-header">📊 Processing Results</div>', unsafe_allow_html=True)
            
            if hasattr(st.session_state, 'processing_results'):
                results = st.session_state.processing_results
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    <div class="success-box">
                        <h4>✅ Processing Complete</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write(f"Frames Processed: {results.get('processed_frames', 'N/A')}")
                    st.write(f"Unique People: {results.get('unique_people', 'N/A')}")
                    st.write(f"Total Detections: {results.get('total_detections', 'N/A')}")
                
                with col2:
                    # Display processed video
                    st.video(st.session_state.processed_video_path)
                    
                    # Download button for processed video
                    with open(st.session_state.processed_video_path, "rb") as file:
                        st.download_button(
                            label="📥 Download Processed Video",
                            data=file,
                            file_name=os.path.basename(st.session_state.processed_video_path),
                            mime="video/mp4",
                            key="download_processed"
                        )
    
    def render_recent_activity(self):
        """Display recent face detection activity from database"""
        st.markdown("---")
        st.markdown('<div class="sub-header">📈 Recent Activity</div>', unsafe_allow_html=True)
        
        try:
            recent_records = self.db_manager.get_visit_records(limit=8)
            
            if recent_records:
                for record in recent_records:
                    timestamp = record.get("timestamp", "")
                    if timestamp:
                        try:
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            display_time = dt.strftime("%H:%M:%S")
                        except:
                            display_time = timestamp
                    
                    # Display activity in columns
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.write(f"**{record.get('person_id', 'Unknown')}**")
                    with col2:
                        action = "🟢 Entry" if record.get("action") == "entry" else "🔴 Exit"
                        st.write(action)
                    with col3:
                        st.write(display_time)
                st.markdown("---")
            else:
                st.info("No activity records")
                
        except Exception as e:
            st.error(f"Error loading activity: {e}")
    
    def render_settings(self):
        """Render system settings and configuration page"""
        st.markdown('<h1 class="main-header">⚙️ System Settings</h1>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Configuration")
            st.json(self.config)
            
            # Configuration download
            config_json = json.dumps(self.config, indent=2)
            st.download_button(
                label="📥 Download Config",
                data=config_json,
                file_name="config.json",
                mime="application/json"
            )
        
        with col2:
            st.markdown("### Database Info")
            try:
                unique_visitors = self.db_manager.get_unique_visitor_count()
                total_records = self.db_manager.get_visit_count("entry") + self.db_manager.get_visit_count("exit")
                
                st.metric("Unique Visitors", unique_visitors)
                st.metric("Total Records", total_records)
                st.write(f"Database: {self.config.get('mongodb_uri', 'Not configured')}")
                
                # Database connection management
                if st.button("🔄 Refresh Connection", key="refresh_db"):
                    try:
                        self.db_manager = DatabaseManager(self.config)
                        st.success("Database refreshed!")
                    except Exception as e:
                        st.error(f"Connection error: {e}")
                
            except Exception as e:
                st.error(f"Database error: {e}")
            
            st.markdown("### System Info")
            st.write(f"Python: {sys.version}")
            st.write(f"Directory: {os.getcwd()}")


def main():
    """Main application entry point"""
    app = StreamlitApp()
    app_mode = app.render_sidebar()
    
    # Route to appropriate page based on navigation
    if app_mode == "🏠 Dashboard":
        app.render_dashboard()
    elif app_mode == "⚙️ Settings":
        app.render_settings()


if __name__ == "__main__":
    main()