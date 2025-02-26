import pymongo
from alfred.utils import MongoConnectionStrings


def drop_model_db():
    """
    Drops the entire 'model_db' database after user confirmation.
    WARNING: This will delete ALL data in the model_db database, not just specific experiments.
    """
    # Connect to MongoDB
    connection = MongoConnectionStrings()
    mongo_client = connection.get_mongo_client()

    # Check if model_db exists
    db_names = mongo_client.list_database_names()

    if 'model_db' not in db_names:
        print("Database 'model_db' does not exist.")
        return

    print("WARNING: You are about to delete the entire 'model_db' database.")
    print("This action cannot be undone and will remove all collections and data in model_db.")

    # Confirm deletion
    confirm = input("Are you sure you want to delete the entire 'model_db' database? (yes/no): ")
    if confirm.lower() == 'yes':
        # Additional confirmation for safety
        final_confirm = input("Please type 'DELETE' to confirm database deletion: ")
        if final_confirm == 'DELETE':
            # Perform deletion
            mongo_client.drop_database('model_db')
            print("Database 'model_db' has been deleted.")
        else:
            print("Deletion aborted - final confirmation failed.")
    else:
        print("Deletion aborted.")


def main():
    print("**************This script will delete the entire 'model_db' database.")
    drop_model_db()


if __name__ == "__main__":
    main()