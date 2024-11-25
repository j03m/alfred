import json

class MongoConnectionStrings:
    def __init__(self, file_name="./db/mongo.json"):
        # Load JSON data from the provided file
        with open(file_name, 'r') as file:
            self.data = json.load(file)

    def connection_string(self):
        return f"mongodb://{self.data['host']}:{self.data['port']}"