from sacred import Experiment
from sacred.observers import MongoObserver
import argparse
import json
from alfred.utils import ExperimentSelector

# Initialize Sacred experiment
ex = Experiment("experiment_runner")
# Attach a MongoDB observer to store experiment results
ex.observers.append(MongoObserver(url='mongodb://localhost:27017', db_name='sacred'))


# Experiment configuration (default values)
@ex.config
def config():
    model = None  # The model type
    size = None  # The model size
    data = None  # Data combination


# Main function to run the experiment
@ex.automain
def run_experiment(model, size, data):
    print(f"Running experiment: Model={model}, Size={size}, Data={data}")
    # Placeholder for the actual experiment logic.
    # You can insert the model training or evaluation code here.
    # For example:
    # results = train_model(model, size, data)
    # ex.log_scalar('accuracy', results['accuracy'])
    return {"model": model, "size": size, "data": data}


def main(index_file, include, exclude):
    # Use ExperimentSelector to select experiments based on ranges
    selector = ExperimentSelector(index_file)
    experiments = selector.get(include_ranges=include, exclude_ranges=exclude)

    # Run the experiments using Sacred
    for experiment in experiments:
        if experiment:
            ex.run(config_updates={
                'model': experiment['model'],
                'size': experiment['size'],
                'data': experiment['data']
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run selected experiments using Sacred.")
    parser.add_argument("--index-file", type=str, default="./metadata/experiment-index.json",
                        help="Path to the JSON file containing indexed experiments")
    parser.add_argument("--include", type=str, default="",
                        help="Ranges of experiments to include (e.g., 1-5,10-15)")
    parser.add_argument("--exclude", type=str, default="",
                        help="Ranges of experiments to exclude (e.g., 4-5,8)")

    args = parser.parse_args()
    main(args.index_file, args.include, args.exclude)
