# src/models/session_models.py
from datetime import datetime
from typing import List, Optional, Dict, Any
from bson import ObjectId
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from pydantic_core import core_schema

class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler) -> core_schema.CoreSchema:
        return core_schema.str_schema()
    
    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

class SessionBase(BaseModel):
    session_id: str
    session_name: str  #for user-friendly names
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now) 
    status: str = "processing"  # processing, completed, failed, cloned, merged
    progress: Dict[str, float] = Field(default_factory=dict)
    coordinate_data: Dict[str, Any]
    original_files: Dict[str, Any]  # Changed from List[str] to Any to support both string and list
    error: Optional[str] = None  # ADD THIS - for storing error messages

class SessionCreate(SessionBase):
    pass

class Session(SessionBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class MarkdownSection(BaseModel):
    session_id: str
    section_id: str
    content: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class ProcessedFile(BaseModel):
    session_id: str
    file_type: str  # ocr, chunked, classified, inception, etc.
    gcs_path: str  
    file_size: int
    created_at: datetime = Field(default_factory=datetime.now)