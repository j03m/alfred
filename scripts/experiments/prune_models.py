from alfred.model_persistence import prune_old_versions

if __name__ == "__main__":
    # Specify the directory containing the model files
    model_directory = './models'  # Update this path if needed
    prune_old_versions(model_directory)