import torch
import io
import gridfs
import joblib
import zlib

from alfred.devices import build_model_token, set_device
from alfred.models import (LSTMModel, LSTMConv1d, AdvancedLSTM, TransAm, Vanilla, LstmConcatExtractors,
                           LstmLayeredExtractors, ExtractorType, TransformerLayeredExtractors)
from alfred.utils import MongoConnectionStrings

import torch.optim as optim
import torch.nn as nn

DEVICE = set_device()

# MongoDB setup
connection = MongoConnectionStrings()
mongo_client = connection.get_mongo_client()
db = mongo_client['model_db']
fs = gridfs.GridFS(db)
models_collection = db['models']
status_collection = db['model_status']
metrics_collection = db['metrics']


def create_encoder_decoder_layers(size):
    sizes = [size]
    while size // 2 != 1:
        size = size // 2
        sizes.append(size)
    return sizes


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
        model = Vanilla(input_size=num_features, hidden_size=size, layers=3, output_size=output)
    elif config_token == 'vanilla.large':
        model = Vanilla(input_size=num_features, hidden_size=size, layers=20, output_size=output)
    elif config_token == 'vanilla.medium':
        model = Vanilla(input_size=num_features, hidden_size=size, layers=10, output_size=output)
    elif config_token == 'vanilla.medium.identity':
        model = Vanilla(input_size=num_features, hidden_size=size, layers=10, output_size=output,
                        final_activation=nn.Identity())
    elif config_token == 'vanilla.small.tanh':
        model = Vanilla(input_size=num_features, hidden_size=size, layers=3, output_size=output,
                        final_activation=nn.Tanh())
    elif config_token == 'vanilla.medium.tanh':
        model = Vanilla(input_size=num_features, hidden_size=size, layers=5, output_size=output,
                        final_activation=nn.Tanh())
    elif config_token == 'vanilla.large.tanh':
        model = Vanilla(input_size=num_features, hidden_size=size, layers=10, output_size=output,
                        final_activation=nn.Tanh())
    elif config_token == 'lstm.medium.extractors.tanh':
        model = LstmConcatExtractors(input_size=num_features, seq_len=sequence_length, hidden_size=size, output_size=output,
                                     extractor_types=[ExtractorType.LSTM, ExtractorType.ATTENTION,
                                                      ExtractorType.CONVOLUTION], final_activation=nn.Tanh(), layers=3)
    elif config_token == 'att.conv.medium.extractors.tanh':
        model = LstmConcatExtractors(input_size=num_features, seq_len=sequence_length, hidden_size=size, output_size=output,
                                     extractor_types=[ExtractorType.ATTENTION,
                                                      ExtractorType.CONVOLUTION], final_activation=nn.Tanh(), layers=3)
    elif config_token == 'lstm.medium.extractors.layered.tanh':
        # same model but order here is important!
        model = LstmLayeredExtractors(input_size=num_features, seq_len=sequence_length, hidden_size=size, output_size=output,
                                      final_activation=nn.Tanh(), layers=3)
    elif config_token == 'trans.medium.extractors.layered.tanh':
        # same model but order here is important!
        model = TransformerLayeredExtractors(input_size=num_features, seq_len=sequence_length, hidden_size=size, output_size=output,
                                             final_activation=nn.Tanh(), layers=3)
    elif config_token == 'trans.medium.extractors.slim.tanh':
        # same model but order here is important!
        model = TransformerLayeredExtractors(input_size=num_features, seq_len=sequence_length, hidden_size=size, output_size=output,
                                             final_activation=nn.Tanh(), layers=3, extra_attention=False)
    else:
        raise Exception("Model type not supported")

    model.to(DEVICE)

    model_token = build_model_token(descriptors)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=100,  # Increased patience to allow for more epochs before adjustment
        factor=0.75,  # Less aggressive reduction
        cooldown=100,  # Wait 3 epochs after reducing before considering another reduction
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
        print(f"\nNew best model: {eval_loss} vs {best_loss}: saving")
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

    print(f"Found scaler version {record['version']} for {model_token}.")

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
    file_buffer_value = buffer.getvalue()
    model_file_id = fs.put(file_buffer_value, filename=f"{model_token}_v{next_version}")

    # --- Serialize Scaler ---
    scaler_buffer = io.BytesIO()
    joblib.dump(scaler, scaler_buffer)  # Serialize scaler to buffer
    scaler_buffer.seek(0)
    scaler_buffer_value = scaler_buffer.getvalue()
    scaler_file_id = fs.put(scaler_buffer_value,
                            filename=f"{model_token}_scaler_v{next_version}")  # Save scaler to GridFS

    # Save to MongoDB
    models_collection.insert_one({
        'model_token': model_token,
        'version': next_version,
        'eval_loss': eval_loss,
        'model_file_id': model_file_id,
        'scaler_file_id': scaler_file_id
    })
    del buffer
    del scaler_buffer
    del scaler_buffer_value
    del file_buffer_value

    print(f"Model version {next_version} saved for {model_token}.")


def prune_old_versions():
    # Group models by their `model_token` and find the latest version for each
    pipeline = [
        {'$sort': {'version': -1}},  # Sort by version in descending order
        {'$group': {
            '_id': '$model_token',  # Group by model_token
            'latest_id': {'$first': '$_id'},  # Get the _id of the latest version
            'latest_model_file_id': {'$first': '$model_file_id'},  # Keep latest model file
            'latest_scaler_file_id': {'$first': '$scaler_file_id'}  # Keep latest scaler file
        }}
    ]

    # Get the list of latest versions to keep
    latest_versions = list(models_collection.aggregate(pipeline))
    latest_ids = {doc['latest_id'] for doc in latest_versions}

    # Find all models to delete (those not in latest_ids) and collect their file_ids
    old_models = models_collection.find({'_id': {'$nin': list(latest_ids)}})
    old_file_ids = []
    for model in old_models:
        if 'model_file_id' in model:
            old_file_ids.append(model['model_file_id'])
        if 'scaler_file_id' in model:
            old_file_ids.append(model['scaler_file_id'])

    # Delete all other versions
    result = models_collection.delete_many({'_id': {'$nin': list(latest_ids)}})

    # Delete associated GridFS files (this also removes chunks)
    deleted_files = 0
    for file_id in old_file_ids:
        try:
            fs.delete(file_id)  # Deletes from fs.files and fs.chunks
            deleted_files += 1
        except gridfs.errors.NoFile:
            print(f"ERROR: GridFS file with ID {file_id} not found, skipping, but this might be a problem")

    print(f"Pruned {result.deleted_count} old model versions.")
