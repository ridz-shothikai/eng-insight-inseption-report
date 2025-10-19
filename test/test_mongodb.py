# test_mongodb.py
import asyncio
import os
from src.database.mongodb import mongodb

async def test_connection():
    try:
        await mongodb.connect()
        print("✅ MongoDB connection successful!")
        
        db = mongodb.client[os.getenv("MONGODB_DB", "rfp_processor")]
        collections = await db.list_collection_names()
        print(f"✅ Available collections: {collections}")
        
        await mongodb.close()
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())