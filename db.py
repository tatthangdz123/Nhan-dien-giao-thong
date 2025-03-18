import pymongo

def setup_database():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["traffic_db"]
    
    # Tạo collection nếu chưa có
    if "violations" not in db.list_collection_names():
        db.create_collection("violations")
        print("Collection 'violations' created successfully!")
    else:
        print("Collection 'violations' already exists.")
    
    client.close()

if __name__ == "__main__":
    setup_database()
