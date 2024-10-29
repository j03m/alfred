import pymongo
import pandas as pd

# MongoDB connection parameters
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client['sacred_db']
collection = db['runs']

# Query to retrieve all 'COMPLETED' experiments
cursor = collection.find({'status': 'COMPLETED'}, {
    '_id': 1,                     # Experiment ID
    'experiment.name': 1,          # Experiment name
    'config': 1,                   # Configuration parameters
    'result': 1,                   # Results
    'status': 1,                   # Status
    'start_time': 1,               # Start time
    'stop_time': 1,                # Stop time
    'info': 1
})

# List to hold each row of the data
data_list = []

# Iterate through the cursor to flatten and extract fields
for doc in cursor:
    row = {}

    # Basic fields
    row['_id'] = str(doc['_id'])  # Convert ObjectId to string for CSV compatibility
    row['start_time'] = doc.get('start_time', None)
    row['stop_time'] = doc.get('stop_time', None)

    # Flatten 'config' field: Add each config parameter as a separate column
    config = doc.get('config', {})
    for key, value in config.items():
        row[f'config_{key}'] = value

    # Flatten 'result' field: Add each result name as a separate column
    result = doc.get('result', {})
    for key, value in result.items():
        if isinstance(value, dict) and 'value' in value:
            row[f'{key}'] = value['value']
        elif isinstance(value, dict):  # If nested dict (e.g., 'ledger_metrics')
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict) and 'value' in sub_value:
                    row[f'{sub_key}'] = sub_value['value']
                else:
                    row[f'{sub_key}'] = sub_value
        else:
            row[f'{key}'] = value

    mt = doc.get('info', {}).get('model_token', "unknown")
    row[f'model_token'] = mt
    # Append the row to the data list
    data_list.append(row)

# Convert the list of rows into a pandas DataFrame
df = pd.DataFrame(data_list)

# Write the DataFrame to a CSV file
df.to_csv('./results/completed_experiments.csv', index=False)

print("Experiments exported to 'completed_experiments.csv'")
