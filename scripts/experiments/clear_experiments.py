import pymongo
import argparse
from alfred.utils import MongoConnectionStrings

def clear_experiments(namespace):
    # Connect to MongoDB
    connection = MongoConnectionStrings()
    mongo_client = connection.get_mongo_client()
    db = mongo_client['sacred_db']
    collection = db['runs']

    # Query to find experiments with the specified namespace
    query = {'experiment.name': namespace}

    # Count the experiments to be deleted
    to_delete_count = collection.count_documents(query)
    print(f"Found {to_delete_count} experiments for namespace '{namespace}'.")

    if to_delete_count > 0:
        # Confirm deletion
        confirm = input(f"Are you sure you want to delete all {to_delete_count} experiments? (yes/no): ")
        if confirm.lower() == 'yes':
            # Perform deletion
            result = collection.delete_many(query)
            print(f"Deleted {result.deleted_count} experiments from the namespace '{namespace}'.")
        else:
            print("Deletion aborted.")
    else:
        print("No experiments found to delete.")

def main():
    parser = argparse.ArgumentParser(description="Clear MongoDB experiments for a specific namespace.")
    parser.add_argument("--token", type=str, required=True, help="Namespace of the experiment set to clear")
    args = parser.parse_args()
    clear_experiments(args.token)

if __name__ == "__main__":
    main()