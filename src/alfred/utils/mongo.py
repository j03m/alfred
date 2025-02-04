import json
import pymongo

_client = None
_data = None
class MongoConnectionStrings:
    def __init__(self, file_name="./db/mongo.json"):
        # Load JSON data from the provided file
        with open(file_name, 'r') as file:
            global _data
            if _data is None:
                _data = json.load(file)

    def connection_string(self):
        global _data
        return f"mongodb://{_data['host']}:{_data['port']}"

    def get_mongo_client(self, timeout=3000):
        global _client, _data
        """
        Attempts to connect to MongoDB using the provided connection string.
        Falls back to localhost if the primary connection fails.
        """
        if _client is None:
            connection = MongoConnectionStrings()
            try:
                # Attempt primary connection
                client = pymongo.MongoClient(connection.connection_string(), serverSelectionTimeoutMS=timeout)
                # Test connection
                client.admin.command("ping")
                print("Connected to MongoDB (primary).")
                _client = client
            except Exception as e:
                print(f"Primary connection failed: {e}")
                try:
                    # Fallback to localhost
                    # Why do we do this? I work on the train alot and it was a
                    # big PIA to keep changing the connection string.
                    client = pymongo.MongoClient("mongodb://localhost:27017/")
                    client.admin.command("ping")
                    _data["host"] = "localhost"
                    _data["port"] = 27017
                    print("Connected to MongoDB (fallback to localhost).")
                    _client = client
                except Exception as fallback_error:
                    print(f"Fallback connection also failed: {fallback_error}")
                    raise RuntimeError("Unable to connect to MongoDB.")
        return _client
