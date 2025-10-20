import os
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
import logging
from dotenv import load_dotenv

# Load .env from project root 
project_root = Path(__file__).parent.parent.parent  # Go up one more level
dotenv_path = project_root / ".env"

# Try multiple possible locations for .env file
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
else:
    # Try current directory as fallback
    load_dotenv()

logger = logging.getLogger(__name__)

class MongoDB:
    def __init__(self):
        self.client = None
        self.sync_client = None
        self.db_name = os.getenv("MONGODB_DB", "rfp_processor")

    async def connect(self):
        mongo_url = os.getenv("MONGODB_URL")
        if not mongo_url:
            # Debug information
            logger.error(f"❌ MONGODB_URL not found. Current working directory: {os.getcwd()}")
            logger.error(f"❌ Project root: {project_root}")
            logger.error(f"❌ .env path attempted: {dotenv_path}")
            logger.error(f"❌ .env exists: {dotenv_path.exists()}")
            raise ValueError("❌ MONGODB_URL is not set in environment or .env file")

        logger.info(f"Connecting to MongoDB DB '{self.db_name}' at {mongo_url}")
        try:
            self.client = AsyncIOMotorClient(
                mongo_url,
                maxPoolSize=10,
                minPoolSize=5
            )
            self.sync_client = MongoClient(mongo_url)

            # Test connection
            await self.client.admin.command('ping')
            logger.info("✅ Connected to MongoDB")
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            raise

    async def close(self):
        if self.client:
            self.client.close()
        if self.sync_client:
            self.sync_client.close()
        logger.info("✅ MongoDB connection closed")

# Global instance
mongodb = MongoDB()

def get_database():
    if mongodb.client is None:
        raise RuntimeError("MongoDB not connected. Call connect() first.")
    return mongodb.client[mongodb.db_name]

def get_sync_database():
    if mongodb.sync_client is None:
        raise RuntimeError("MongoDB sync client not connected.")
    return mongodb.sync_client[mongodb.db_name]