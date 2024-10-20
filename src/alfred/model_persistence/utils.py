import os
import glob
import torch
import json

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
