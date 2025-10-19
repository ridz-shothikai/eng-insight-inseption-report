import os
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import logging

logger = logging.getLogger(__name__)

class MongoDB:
    client: AsyncIOMotorClient = None
    sync_client: MongoClient = None
    
    async def connect(self):
        """Connect to MongoDB"""
        try:
            # For async operations
            self.client = AsyncIOMotorClient(
                os.getenv("MONGODB_URL", ""),
                maxPoolSize=10,
                minPoolSize=5
            )
            
            # For sync operations (if needed)
            self.sync_client = MongoClient(
                os.getenv("MONGODB_URL", "")
            )
            
            # Test connection
            await self.client.admin.command('ping')
            logger.info("✅ Connected to MongoDB")
            
        except ConnectionFailure as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            raise

    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
        if self.sync_client:
            self.sync_client.close()
        logger.info("✅ MongoDB connection closed")

# Global instance
mongodb = MongoDB()

def get_database():
    """Get database instance"""
    return mongodb.client.get_database(os.getenv("MONGODB_DB", "rfp_processor"))

def get_sync_database():
    """Get sync database instance"""
    return mongodb.sync_client.get_database(os.getenv("MONGODB_DB", "rfp_processor"))