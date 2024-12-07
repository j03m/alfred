import json
import pymongo

class MongoConnectionStrings:
    def __init__(self, file_name="./db/mongo.json"):
        # Load JSON data from the provided file
        with open(file_name, 'r') as file:
            self.data = json.load(file)

    def connection_string(self):
        return f"mongodb://{self.data['host']}:{self.data['port']}"

    def get_mongo_client(self, timeout=3000):
        """
        Attempts to connect to MongoDB using the provided connection string.
        Falls back to localhost if the primary connection fails.
        """
        connection = MongoConnectionStrings()
        try:
            # Attempt primary connection
            client = pymongo.MongoClient(connection.connection_string(), serverSelectionTimeoutMS=timeout)
            # Test connection
            client.admin.command("ping")
            print("Connected to MongoDB (primary).")
            return client
        except Exception as e:
            print(f"Primary connection failed: {e}")
            try:
                # Fallback to localhost
                client = pymongo.MongoClient("mongodb://localhost:27017/")
                client.admin.command("ping")
                print("Connected to MongoDB (fallback to localhost).")
                return client
            except Exception as fallback_error:
                print(f"Fallback connection also failed: {fallback_error}")
                raise RuntimeError("Unable to connect to MongoDB.")