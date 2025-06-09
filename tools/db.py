from pymongo import MongoClient

class MongoDatabase:
    def __init__(self, 
                 uri: str,
                 db_name: str = 'face_detection_db', 
                 collection_name:str = 'detections'
            ) -> None:
        
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        
        print(f"Connected to MongoDB at {uri}, using database '{db_name}' and collection '{collection_name}'")
        
    def insert_one(self, result:dict)->None:
        result["distance"] = float(result["distance"])
        
        self.collection.insert_one(result)
        
        print(f"Inserted: {result.get('class', 'Unknown')} with distance {result.get('distance', 'N/A')}")