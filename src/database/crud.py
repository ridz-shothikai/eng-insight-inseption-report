from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from bson import ObjectId
import logging
from src.database.mongodb import get_database
from src.models.session_models import Session, MarkdownSection, ProcessedFile

logger = logging.getLogger(__name__)

class SessionCRUD:
    def __init__(self):
        self._db = None
        self._sessions = None
    
    @property
    def db(self):
        if self._db is None:
            self._db = get_database()
        return self._db
    
    @property
    def sessions(self):
        if self._sessions is None:
            self._sessions = self.db.sessions
        return self._sessions
    
    async def create_session(self, session_data: dict) -> str:
        """Create a new session"""
        result = await self.sessions.insert_one(session_data)
        return str(result.inserted_id)
    
    async def get_session(self, session_id: str) -> Optional[dict]:
        """Get session by session_id"""
        return await self.sessions.find_one({"session_id": session_id})
    
    async def update_session_progress(self, session_id: str, progress: Dict[str, float]):
        """Update session progress"""
        await self.sessions.update_one(
            {"session_id": session_id},
            {"$set": {"progress": progress, "updated_at": datetime.now()}}
        )
    
    async def update_session_status(self, session_id: str, status: str, error: str = None):
        """Update session status"""
        update_data = {"status": status, "updated_at": datetime.now()}
        if error:
            update_data["error"] = error
        
        await self.sessions.update_one(
            {"session_id": session_id},
            {"$set": update_data}
        )
    
    async def get_all_sessions(self, limit: int = 100) -> List[dict]:
        """Get all sessions"""
        cursor = self.sessions.find().sort("created_at", -1).limit(limit)
        return await cursor.to_list(length=limit)
    
    async def delete_session(self, session_id: str):
        """Delete a session"""
        await self.sessions.delete_one({"session_id": session_id})

class MarkdownCRUD:
    def __init__(self):
        self._db = None
        self._markdown_sections = None
    
    @property
    def db(self):
        if self._db is None:
            self._db = get_database()
        return self._db
    
    @property
    def markdown_sections(self):
        if self._markdown_sections is None:
            self._markdown_sections = self.db.markdown_sections
        return self._markdown_sections
    
    async def save_markdown_section(self, session_id: str, section_id: str, content: str):
        """Save or update a markdown section"""
        await self.markdown_sections.update_one(
            {"session_id": session_id, "section_id": section_id},
            {
                "$set": {
                    "content": content,
                    "updated_at": datetime.now()
                },
                "$setOnInsert": {
                    "created_at": datetime.now()
                }
            },
            upsert=True
        )
    
    async def get_session_markdown(self, session_id: str) -> Dict[str, str]:
        """Get all markdown sections for a session"""
        cursor = self.markdown_sections.find({"session_id": session_id})
        sections = await cursor.to_list(length=None)
        return {section["section_id"]: section["content"] for section in sections}
    
    async def delete_session_markdown(self, session_id: str):
        """Delete all markdown sections for a session"""
        await self.markdown_sections.delete_many({"session_id": session_id})

class FileCRUD:
    def __init__(self):
        self._db = None
        self._processed_files = None
    
    @property
    def db(self):
        if self._db is None:
            self._db = get_database()
        return self._db
    
    @property
    def processed_files(self):
        if self._processed_files is None:
            self._processed_files = self.db.processed_files
        return self._processed_files
    
    async def save_file_metadata(self, session_id: str, file_type: str, gcs_path: str, file_size: int):
        """Save processed file metadata"""
        await self.processed_files.insert_one({
            "session_id": session_id,
            "file_type": file_type,
            "gcs_path": gcs_path,  # Store GCS path instead of local path
            "file_size": file_size,
            "created_at": datetime.now()
        })
    
    async def get_session_files(self, session_id: str) -> List[dict]:
        """Get all files for a session"""
        cursor = self.processed_files.find({"session_id": session_id})
        return await cursor.to_list(length=None)
    
    async def delete_session_files(self, session_id: str):
        """Delete all file metadata for a session"""
        await self.processed_files.delete_many({"session_id": session_id})

# Global instances - these will initialize connections when first used
session_crud = SessionCRUD()
markdown_crud = MarkdownCRUD()
file_crud = FileCRUD()