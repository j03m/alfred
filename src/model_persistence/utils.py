import os
import glob
import torch
import json


def maybe_save_model(epoch, evaluator, eval_save, model, model_path, model_prefix):
    if eval_save:
        eval_loss = evaluator()
        best_loss = get_best_loss(model_path, model_prefix)
        if eval_loss < best_loss:
            print(f"New best model: {eval_loss} vs {best_loss}: saving")
            save_next_model(model, model_path, model_prefix)
            set_best_loss(model_path, model_prefix, eval_loss)
        else:
            print(f"{eval_loss} vs {best_loss}: declining save")
    else:
        print("saving model at: ", epoch)
        save_next_model(model, model_path, model_prefix)


def get_best_loss(model_path, model_prefix):
    try:
        with open(f"{model_path}/{model_prefix}-metrics.json", 'r') as f:
            metrics = json.load(f)
        best_loss = metrics['best_loss']
    except (FileNotFoundError, KeyError):
        best_loss = float('inf')
    return best_loss


def set_best_loss(model_path, model_prefix, loss):
    with open(f"{model_path}/{model_prefix}-metrics.json", 'w') as f:
        json.dump({'best_loss': loss}, f)


def get_latest_model(model_path, model_prefix):
    search_pattern = os.path.join(model_path, f"{model_prefix}*.pth")
    model_files = glob.glob(search_pattern)
    if not model_files:
        print("No previous model.")
        return None
    model_files.sort(key=lambda x: int(os.path.basename(x)[len(model_prefix):-4]), reverse=True)
    print(f"Found {model_files[0]} for previous model.")
    return model_files[0]


def save_next_model(model, model_path, model_prefix):
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
    torch.save(model.state_dict(), next_model_path)
    print(f"Model saved to {next_model_path}")
    return next_model_path
