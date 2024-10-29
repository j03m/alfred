import os
import glob
import torch
import json
import re


def maybe_save_model_with_evaluator(epoch, evaluator, eval_save, model, model_path, model_prefix):
    if eval_save:
        eval_loss = evaluator()
        return maybe_save_model(model, eval_loss, model_path, model_prefix)
    else:
        print("saving model at: ", epoch)
        save_next_model(model, model_path, model_prefix)
        return True


def maybe_save_model(model, optimizer, scheduler, eval_loss, model_path, model_prefix, training_label="all"):
    best_loss = get_best_loss(model_path, model_prefix, training_label)
    if eval_loss < best_loss:
        print(f"New best model: {eval_loss} vs {best_loss}: saving")
        save_next_model(model, optimizer, scheduler, model_path, model_prefix)
        set_best_loss(model_path, model_prefix, eval_loss, training_label)
        return True
    else:
        print(f"{eval_loss} vs {best_loss}: declining save")
        return False


def get_best_loss(model_path, model_prefix, token="all"):
    try:
        with open(f"{model_path}/{model_prefix}-{token}-metrics.json", 'r') as f:
            metrics = json.load(f)
        best_loss = metrics['best_loss']
    except (FileNotFoundError, KeyError):
        best_loss = float('inf')
    return best_loss


def set_best_loss(model_path, model_prefix, loss, token="all"):
    with open(f"{model_path}/{model_prefix}-{token}-metrics.json", 'w') as f:
        json.dump({'best_loss': loss}, f)


def get_latest_model(model_path, model_prefix):
    search_pattern = os.path.join(model_path, f"{model_prefix}*.pth")
    model_files = glob.glob(search_pattern)
    if not model_files:
        print("No previous model.")
        return None
    model_files.sort(key=lambda x: int(os.path.basename(x)[len(model_prefix):-4]), reverse=True)
    print(f"Found {model_files[0]} for previous model.")
    latest_model_path = model_files[0]
    return torch.load(latest_model_path)


def save_next_model(model, optimizer, scheduler, model_path, model_prefix):
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    search_pattern = os.path.join(model_path, f"{model_prefix}*.pth")

    model_files = glob.glob(search_pattern)

    max_counter = -1
    for model_file in model_files:
        basename = os.path.basename(model_file)
        counter = basename[len(model_prefix):-4]
        try:
            counter = int(counter)
            if counter > max_counter:
                max_counter = counter
        except ValueError:
            continue  # Skip files that do not end with a number

    next_counter = max_counter + 1
    next_model_filename = f"{model_prefix}{next_counter}.pth"
    next_model_path = os.path.join(model_path, next_model_filename)

    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
    }, next_model_path)
    print(f"Model saved to {next_model_path}")

    return next_model_path

def prune_old_versions(directory):
    # Updated regex pattern to match the device token dynamically (cpu, cuda, mps, etc.)
    # Matches the format lstm_30_128_1_0x216c6ee9_<device><number>.pth
    pattern = re.compile(r'(.*_)(cpu|cuda|mps)(\d+)\.pth')

    # Dictionary to store the latest version of each model based on the device token
    latest_versions = {}

    # Traverse the files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith('.pth'):
            match = pattern.match(filename)
            if match:
                model_name = match.group(1)  # The part before the device token
                device_token = match.group(2)  # The device token (cpu, cuda, mps)
                version = int(match.group(3))  # The version number

                # Create a unique key for each model+device combination
                model_device_key = f"{model_name}{device_token}"

                # Check if it's the latest version for this model+device combination
                if model_device_key not in latest_versions or version > latest_versions[model_device_key][1]:
                    latest_versions[model_device_key] = (filename, version)

    # Now we can delete the older versions
    for filename in os.listdir(directory):
        if filename.endswith('.pth'):
            match = pattern.match(filename)
            if match:
                model_name = match.group(1)
                device_token = match.group(2)
                version = int(match.group(3))

                # Create the key again to check if this is the latest version
                model_device_key = f"{model_name}{device_token}"

                # If this is not the latest version, delete the file
                if version != latest_versions[model_device_key][1]:
                    file_path = os.path.join(directory, filename)
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")


