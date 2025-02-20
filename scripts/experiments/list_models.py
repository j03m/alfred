import pymongo
from alfred.utils import MongoConnectionStrings


def list_models():
    """
    Lists all collections (models) in the 'model_db' database.
    """
    # Connect to MongoDB
    connection = MongoConnectionStrings()
    mongo_client = connection.get_mongo_client()

    # Access the 'model_db' database
    model_db = mongo_client['model_db']

    # Get list of collection names (model names)
    model_names = model_db.list_collection_names()

    if model_names:
        print("Models in 'model_db' database:")
        for model_name in model_names:
            print(f"- {model_name}")
    else:
        print("No models found in 'model_db' database.")


def main():
    print("Listing models in 'model_db' database:")
    list_models()


if __name__ == "__main__":
    main()