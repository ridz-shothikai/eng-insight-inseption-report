# test_mongodb.py
import os
import sys
import asyncio

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.mongodb import mongodb
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_connection():
    try:
        print("🔗 Attempting to connect to MongoDB...")
        await mongodb.connect()
        print("✅ MongoDB connection successful!")
        
        # Test database operations
        db = mongodb.client[mongodb.db_name]
        collections = await db.list_collection_names()
        print(f"✅ Available collections: {collections}")
        
        # Test creating a sample document
        test_collection = db.test_connection
        result = await test_collection.insert_one({"test": "connection", "timestamp": "now"})
        print(f"✅ Test document inserted with ID: {result.inserted_id}")
        
        # Clean up
        await test_collection.delete_one({"_id": result.inserted_id})
        print("✅ Test document cleaned up")
        
        await mongodb.close()
        print("✅ MongoDB connection closed properly")
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_connection())