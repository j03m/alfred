import pymongo
from alfred.utils import MongoConnectionStrings
import sys
import time

def drop_model_db(force=False):
    """
    Drops the entire 'model_db' database.
    If force is False (default), prompts the user for confirmation.
    If force is True, performs a 5-second countdown before deleting the database.
    WARNING: This will delete ALL data in the model_db database.
    """
    # Connect to MongoDB
    connection = MongoConnectionStrings()
    mongo_client = connection.get_mongo_client()

    # Check if model_db exists
    db_names = mongo_client.list_database_names()

    if 'model_db' not in db_names:
        print("Database 'model_db' does not exist.")
        return

    if force:
        print("WARNING: Force flag is set. Proceeding with countdown to delete 'model_db' database.")
        try:
            for i in range(5, 0, -1):
                print(f"Counting down from {i} - bro are you sure you want to wipe the model db?", flush=True)
                time.sleep(1)
            print("Proceeding to delete the database...")
            mongo_client.drop_database('model_db')
            print("Database 'model_db' has been deleted.")
        except KeyboardInterrupt:
            print("\nDeletion aborted by user.")
    else:
        print("WARNING: You are about to delete the entire 'model_db' database.")
        print("This action cannot be undone and will remove all collections and data in model_db.")
        confirm = input("Are you sure you want to delete the entire 'model_db' database? (yes/no): ")
        if confirm.lower() == 'yes':
            final_confirm = input("Please type 'DELETE' to confirm database deletion: ")
            if final_confirm == 'DELETE':
                mongo_client.drop_database('model_db')
                print("Database 'model_db' has been deleted.")
            else:
                print("Deletion aborted - final confirmation failed.")
        else:
            print("Deletion aborted.")

def main():
    print("**************This script will delete the entire 'model_db' database.")
    force = "-f" in sys.argv
    drop_model_db(force)

if __name__ == "__main__":
    main()