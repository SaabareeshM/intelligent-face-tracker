import logging
import numpy as np
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import uuid

logger = logging.getLogger("face_pipeline")


class DatabaseManager:
    """MongoDB manager for face tracking data storage and retrieval"""
    
    def __init__(self, config):
        self.config = config
        self.db = self._init_mongodb()
        self._init_counter()
    
    def _init_mongodb(self):
        """Initialize MongoDB connection and ensure required collections exist"""
        try:
            mongodb_uri = self.config.get("mongodb_uri", "mongodb://localhost:27017/")
            client = MongoClient(mongodb_uri)
            client.admin.command('ping')  # Test connection
            db_name = self.config.get("database_name", "face_tracker")
            db = client[db_name]
            
            # Ensure required collections exist
            collections = db.list_collection_names()
            required = ['people', 'face_data', 'visit_records', 'counter']
            
            for coll in required:
                if coll not in collections:
                    db.create_collection(coll)
                    logger.info(f"Created collection: {coll}")
            
            # Create indexes for performance
            db.people.create_index("person_id", unique=True)
            db.face_data.create_index("person_id")
            db.visit_records.create_index("person_id")
            db.visit_records.create_index("action")
            db.visit_records.create_index("timestamp")
            
            logger.info(f"Connected to MongoDB: {db_name}")
            return db
        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise
    
    def _init_counter(self):
        """Initialize person counter for sequential ID generation"""
        try:
            counter = self.db.counter.find_one({"name": "person_counter"})
            if not counter:
                self.db.counter.insert_one({"name": "person_counter", "current_number": 0})
                logger.info("Person counter initialized")
        except Exception as e:
            logger.error(f"Counter init error: {e}")
    
    def get_next_person_id(self):
        """Generate next sequential person ID"""
        try:
            result = self.db.counter.find_one_and_update(
                {"name": "person_counter"},
                {"$inc": {"current_number": 1}},
                return_document=True
            )
            return f"person{result['current_number']}"
        except Exception as e:
            logger.error(f"ID generation error: {e}")
            return f"person_{uuid.uuid4().hex[:8]}"  # Fallback to UUID
    
    def cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors"""
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def find_best_match(self, new_embedding, known_embeddings):
        
        best_sim = -1
        best_id = None
        second_best_sim = -1
        
        for person_id, known_emb in known_embeddings:
            sim = self.cosine_similarity(new_embedding, known_emb)
            if sim > best_sim:
                second_best_sim = best_sim
                best_sim = sim
                best_id = person_id
            elif sim > second_best_sim:
                second_best_sim = sim
        
        return best_id, best_sim, second_best_sim
    
    def should_store_embedding(self, person_id, new_embedding):
        
        try:
            # Get recent embeddings for this person
            existing = list(self.db.face_data.find(
                {"person_id": person_id}, 
                {"face_vector": 1}
            ).sort("created_time", -1).limit(10))
            
            if not existing:
                return True  # No existing embeddings, store this one
            
            # Find maximum similarity with existing embeddings
            max_sim = 0
            for emb in existing:
                existing_vec = np.array(emb["face_vector"], dtype=np.float32)
                sim = self.cosine_similarity(new_embedding, existing_vec)
                max_sim = max(max_sim, sim)
            
            # Only store if significantly different
            threshold = self.config.get("embedding_diversity_threshold", 0.85)
            should_store = max_sim < threshold
            
            if should_store:
                logger.debug(f"Storing diverse embedding for {person_id}")
            return should_store
            
        except Exception as e:
            logger.error(f"Embedding check error: {e}")
            return True  # Store by default on error
    
    def register_person(self, person_id, embedding_vector, timestamp_iso):
        """Register a new person in the database"""
        try:
            self.db.people.insert_one({
                "person_id": person_id,
                "first_seen": timestamp_iso,
                "last_seen": timestamp_iso,
                "visit_count": 1
            })
            
            self.db.face_data.insert_one({
                "person_id": person_id,
                "face_vector": embedding_vector.tolist(),
                "created_time": timestamp_iso
            })
            
            logger.debug(f"Registered person {person_id}")
        except Exception as e:
            logger.error(f"Registration error: {e}")
    
    def update_last_seen(self, person_id, timestamp_iso):
        """Update last seen timestamp and increment visit count"""
        try:
            self.db.people.update_one(
                {"person_id": person_id},
                {
                    "$set": {"last_seen": timestamp_iso},
                    "$inc": {"visit_count": 1}
                }
            )
        except Exception as e:
            logger.error(f"Update error for {person_id}: {e}")
    
    def save_visit_record(self, person_id, action, timestamp_iso, img_path):
        """Save visit record (entry or exit) with timestamp"""
        try:
            self.db.visit_records.insert_one({
                "person_id": person_id,
                "action": action,  # 'entry' or 'exit'
                "timestamp": timestamp_iso,
                "photo_path": img_path
            })
        except Exception as e:
            logger.error(f"Visit record error: {e}")
    
    def get_all_face_data(self):
        """Retrieve all face embeddings from database"""
        try:
            cursor = self.db.face_data.find({}, {"person_id": 1, "face_vector": 1})
            result = []
            for doc in cursor:
                vec = np.array(doc["face_vector"], dtype=np.float32)
                result.append((doc["person_id"], vec))
            return result
        except Exception as e:
            logger.error(f"Face data retrieval error: {e}")
            return []
    
    def get_unique_visitor_count(self):
        """Get count of unique visitors"""
        try:
            return self.db.people.count_documents({})
        except Exception as e:
            logger.error(f"Visitor count error: {e}")
            return 0
    
    def get_visit_count(self, action):
        """Get count of specific visit actions (entry/exit)"""
        try:
            return self.db.visit_records.count_documents({"action": action})
        except Exception as e:
            logger.error(f"Visit count error: {e}")
            return 0
        
    def get_people_data(self):
        """Get all people data for analytics"""
        try:
            return list(self.db.people.find())
        except Exception as e:
            logger.error(f"People data error: {e}")
            return []

    def get_visit_records(self, limit=1000):
        """Get recent visit records sorted by timestamp"""
        try:
            return list(self.db.visit_records.find().sort("timestamp", -1).limit(limit))
        except Exception as e:
            logger.error(f"Visit records error: {e}")
            return []