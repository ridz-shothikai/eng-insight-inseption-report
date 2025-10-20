# src/utils/gcs_handler.py

from google.cloud import storage
from pathlib import Path
from typing import List, BinaryIO
import os
from dotenv import load_dotenv
import logging

# Get the project root (2 levels up from this file: root/src/utils/ -> root/)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load .env from project root
load_dotenv(PROJECT_ROOT / '.env')

logger = logging.getLogger(__name__)

class GCSHandler:
    def __init__(self, bucket_name: str = None):
        """Initialize GCS client and bucket"""
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if credentials_path and not os.path.isabs(credentials_path):
            # Resolve relative to project root, not current working directory
            credentials_path = str(PROJECT_ROOT / credentials_path)
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        
        self.client = storage.Client()
        
        # ✅ Read bucket name from .env if not provided
        self.bucket_name = bucket_name or os.getenv('GCS_BUCKET_NAME', 'eng-inception')
        
        self.bucket = self.client.bucket(self.bucket_name)
        logger.info(f"✅ GCS Handler initialized for bucket: {self.bucket_name}")
    
    def upload_file(self, local_path: str, gcs_path: str) -> str:
        """Upload a file to GCS and return the GCS URI"""
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        logger.info(f"📤 Uploaded: {local_path} -> gs://{self.bucket_name}/{gcs_path}")
        return f"gs://{self.bucket_name}/{gcs_path}"
    
    def upload_fileobj(self, file_obj: BinaryIO, gcs_path: str) -> str:
        """Upload a file object to GCS"""
        blob = self.bucket.blob(gcs_path)
        file_obj.seek(0)
        blob.upload_from_file(file_obj)
        logger.info(f"📤 Uploaded file object -> gs://{self.bucket_name}/{gcs_path}")
        return f"gs://{self.bucket_name}/{gcs_path}"
    
    def download_file(self, gcs_path: str, local_path: str) -> str:
        """Download a file from GCS to local path"""
        blob = self.bucket.blob(gcs_path)
        
        # Create parent directory if it doesn't exist
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        
        blob.download_to_filename(local_path)
        logger.info(f"📥 Downloaded: gs://{self.bucket_name}/{gcs_path} -> {local_path}")
        return local_path
    
    def download_to_memory(self, gcs_path: str) -> bytes:
        """Download a file from GCS to memory"""
        blob = self.bucket.blob(gcs_path)
        return blob.download_as_bytes()
    
    def list_files(self, prefix: str = "") -> List[str]:
        """List all files with a given prefix"""
        blobs = self.bucket.list_blobs(prefix=prefix)
        return [blob.name for blob in blobs]
    
    def delete_file(self, gcs_path: str):
        """Delete a file from GCS"""
        blob = self.bucket.blob(gcs_path)
        blob.delete()
        logger.info(f"🗑️ Deleted: gs://{self.bucket_name}/{gcs_path}")
    
    def delete_folder(self, prefix: str):
        """Delete all files with a given prefix (folder)"""
        blobs = self.bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            blob.delete()
        logger.info(f"🗑️ Deleted folder: gs://{self.bucket_name}/{prefix}")
    
    def file_exists(self, gcs_path: str) -> bool:
        """Check if a file exists in GCS"""
        blob = self.bucket.blob(gcs_path)
        return blob.exists()
    
    def get_public_url(self, gcs_path: str) -> str:
        """Get public URL for a file"""
        return f"https://storage.googleapis.com/{self.bucket_name}/{gcs_path}"
    
    def get_signed_url(self, gcs_path: str, expiration: int = 3600) -> str:
        """Generate a signed URL for temporary access"""
        from datetime import timedelta
        blob = self.bucket.blob(gcs_path)
        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(seconds=expiration),
            method="GET"
        )
        return url

# Create singleton instance
gcs_handler = GCSHandler()