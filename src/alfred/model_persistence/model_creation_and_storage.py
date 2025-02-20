import torch
import io
import gridfs
import joblib
import zlib

from alfred.devices import build_model_token, set_device
from alfred.models import LSTMModel, LSTMConv1d, AdvancedLSTM, TransAm, Vanilla
from alfred.utils import MongoConnectionStrings

import torch.optim as optim

DEVICE = set_device()

# MongoDB setup
connection = MongoConnectionStrings()
mongo_client = connection.get_mongo_client()
db = mongo_client['model_db']
fs = gridfs.GridFS(db)
models_collection = db['models']
status_collection = db['model_status']
metrics_collection = db['metrics']


def crc32_columns(strings):
    # Sort the array of strings
    sorted_strings = sorted(strings)

    # Concatenate the sorted strings into one string
    concatenated_string = ''.join(sorted_strings)

    # Convert the concatenated string to bytes
    concatenated_bytes = concatenated_string.encode('utf-8')

    # Compute the CRC32 hash
    crc32_hash = zlib.crc32(concatenated_bytes)

    # Return the hash in hexadecimal format
    return f"{crc32_hash:#08x}"


def eval_model_selector(model_descriptor, column_selector):
    model_token = model_descriptor["model_token"]
    sequence_length = model_descriptor["sequence_length"]
    data = model_descriptor["data"]  # columns
    bar_type = model_descriptor["bar_type"]
    size = model_descriptor["size"]
    columns = sorted(column_selector.get(data))

    output = 1
    model, _, _, real_model_token = model_from_config(
        num_features=len(columns),
        config_token=model_token,
        sequence_length=sequence_length, size=size, output=output,
        descriptors=[
            model_token, sequence_length, size, output, crc32_columns(columns), bar_type
        ])
    return model, real_model_token, columns


def model_from_config(config_token, num_features, sequence_length, size, output, descriptors, layers=2):
    if config_token == 'lstm':
        model = LSTMModel(features=num_features, hidden_dim=size, output_size=output, num_layers=layers)
    elif config_token == 'advanced-lstm':
        model = AdvancedLSTM(features=num_features, hidden_dim=size, output_dim=output)
    elif config_token == 'lstm-conv1d':
        model = LSTMConv1d(features=num_features, seq_len=sequence_length, hidden_dim=size, output_size=output,
                           kernel_size=10)
    elif config_token == 'trans-am':
        model = TransAm(features=num_features, model_size=size, heads=size / 16, output=output, last_bar=True)
    elif config_token == 'vanilla.small':
        model = Vanilla(input_size=num_features, hidden_size=size, output_size=output)
    elif config_token == 'vanilla.large':
        model = Vanilla(input_size=num_features, hidden_size=size, output_size=output, layers=100)
    elif config_token == 'vanilla.medium':
        model = Vanilla(input_size=num_features, hidden_size=size, output_size=output, layers=10)
    else:
        raise Exception("Model type not supported")

    model.to(DEVICE)

    model_token = build_model_token(descriptors)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=10,  # Increased patience to allow for more epochs before adjustment
        factor=0.3,  # Less aggressive reduction
        cooldown=3,  # Wait 3 epochs after reducing before considering another reduction
    )

    # Load the latest model from MongoDB
    model_checkpoint = get_latest_model(model_token)

    if model_checkpoint:
        model.load_state_dict(model_checkpoint['model_state_dict'])
        optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(model_checkpoint['scheduler_state_dict'])
        scaler = get_model_scaler(model_token)
        return model, optimizer, scheduler, scaler, model_token, True
    else:
        return model, optimizer, scheduler, None, model_token, False

def maybe_save_model(model, optimizer, scheduler, scaler, eval_loss, model_token, training_label):
    best_loss = get_best_loss(model_token, training_label)
    if eval_loss < best_loss:
        print(f"New best model: {eval_loss} vs {best_loss}: saving")
        save_next_model(model, optimizer, scheduler, scaler, model_token, eval_loss)
        set_best_loss(model_token, training_label, eval_loss)
        return True
    else:
        return False


def get_best_loss(model_token, training_label):
    record = metrics_collection.find_one({'model_token': model_token, 'training_label': training_label})
    return record['best_loss'] if record else float('inf')


def set_best_loss(model_token, training_label, loss):
    metrics_collection.update_one(
        {'model_token': model_token, 'training_label': training_label},
        {'$set': {'best_loss': loss}},
        upsert=True
    )


# make a record that we already trained against this ticker
def track_model_status(model_token, ticker):
    status_collection.insert_one({
        'status': True,
        'model_token': model_token,
        'ticker': ticker
    })


def check_model_status(model_token, ticker):
    record = status_collection.find_one({'model_token': model_token, 'ticker': ticker})
    if not record:
        return False
    else:
        return True

def get_model_scaler(model_token):
    record = models_collection.find_one({'model_token': model_token}, sort=[('version', -1)])
    if not record:
        print("No previous model.")
        return None

    print(f"Found model version {record['version']} for {model_token}.")

    scaler_file_id = record['scaler_file_id']
    scaler_file = fs.get(scaler_file_id)
    return joblib.load(io.BytesIO(scaler_file.read()))

def get_latest_model(model_token):
    print("Looking for:", model_token)
    record = models_collection.find_one({'model_token': model_token}, sort=[('version', -1)])
    if not record:
        print("No previous model.")
        return None

    print(f"Found model version {record['version']} for {model_token}.")

    file_id = record['model_file_id']
    model_data = fs.get(file_id).read()
    buffer = io.BytesIO(model_data)
    return torch.load(buffer)


def save_next_model(model, optimizer, scheduler, scaler, model_token, eval_loss):
    # Determine the next version number
    last_record = models_collection.find_one({'model_token': model_token}, sort=[('version', -1)])
    next_version = (last_record['version'] + 1) if last_record else 0

    # --- Serialize Model, Optimizer, Scheduler ---
    buffer = io.BytesIO()
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, buffer)
    buffer.seek(0)

    # needed due to models being > bson limit (laaaaame)
    model_file_id = fs.put(buffer.getvalue(), filename=f"{model_token}_v{next_version}")

    # --- Serialize Scaler ---
    scaler_buffer = io.BytesIO()
    joblib.dump(scaler, scaler_buffer)  # Serialize scaler to buffer
    scaler_buffer.seek(0)
    scaler_file_id = fs.put(scaler_buffer.getvalue(),
                            filename=f"{model_token}_scaler_v{next_version}")  # Save scaler to GridFS

    # Save to MongoDB
    models_collection.insert_one({
        'model_token': model_token,
        'version': next_version,
        'eval_loss': eval_loss,
        'model_file_id': model_file_id,
        'scaler_file_id': scaler_file_id
    })
    print(f"Model version {next_version} saved for {model_token}.")


def prune_old_versions():
    # Group models by their `model_token` and find the latest version for each
    pipeline = [
        {'$sort': {'version': -1}},  # Sort by version in descending order
        {'$group': {
            '_id': '$model_token',  # Group by model_token
            'latest_id': {'$first': '$_id'}  # Get the _id of the latest version
        }}
    ]

    # Get the list of latest versions to keep
    latest_versions = list(models_collection.aggregate(pipeline))
    latest_ids = {doc['latest_id'] for doc in latest_versions}

    # Delete all other versions
    result = models_collection.delete_many({'_id': {'$nin': list(latest_ids)}})
    print(f"Pruned {result.deleted_count} old model versions.")
