import torch
import pymongo
import io
import gridfs
from alfred.devices import build_model_token, set_device
from alfred.models import LSTMModel, LSTMConv1d, AdvancedLSTM, TransAm
from alfred.utils import MongoConnectionStrings
import torch.optim as optim

DEVICE = set_device()

# MongoDB setup
connection = MongoConnectionStrings()
mongo_client = pymongo.MongoClient(connection.connection_string())
db = mongo_client['model_db']
fs = gridfs.GridFS(db)
models_collection = db['models']
metrics_collection = db['metrics']


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
    else:
        raise Exception("Model type not supported")

    model.to(DEVICE)

    model_token = build_model_token(descriptors)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # Load the latest model from MongoDB
    model_checkpoint = get_latest_model(model_token)
    if model_checkpoint:
        model.load_state_dict(model_checkpoint['model_state_dict'])
        optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(model_checkpoint['scheduler_state_dict'])

    return model, optimizer, scheduler, model_token


def maybe_save_model(model, optimizer, scheduler, eval_loss, model_token, training_label):
    best_loss = get_best_loss(model_token, training_label)
    if eval_loss < best_loss:
        print(f"New best model: {eval_loss} vs {best_loss}: saving")
        save_next_model(model, optimizer, scheduler, model_token, eval_loss)
        set_best_loss(model_token, training_label, eval_loss)
        return True
    else:
        print(f"{eval_loss} vs {best_loss}: declining save")
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


def get_latest_model(model_token):
    record = models_collection.find_one({'model_token': model_token}, sort=[('version', -1)])
    if not record:
        print("No previous model.")
        return None

    print(f"Found model version {record['version']} for {model_token}.")

    file_id = record['file_id']
    model_data = fs.get(file_id).read()
    buffer = io.BytesIO(model_data)
    return torch.load(buffer)


def save_next_model(model, optimizer, scheduler, model_token, eval_loss):
    # Determine the next version number
    last_record = models_collection.find_one({'model_token': model_token}, sort=[('version', -1)])
    next_version = (last_record['version'] + 1) if last_record else 0

    # Serialize model to binary
    buffer = io.BytesIO()
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, buffer)
    buffer.seek(0)

    # needed due to models being > bson limit (laaaaame)
    file_id = fs.put(buffer.getvalue(), filename=f"{model_token}_v{next_version}")

    # Save to MongoDB
    models_collection.insert_one({
        'model_token': model_token,
        'version': next_version,
        'eval_loss': eval_loss,
        'file_id': file_id
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