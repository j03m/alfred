from itertools import product
import json
import argparse


def main(file, output):
    with open(file, 'r') as file:
        experiment_descriptor = json.load(file)

    keys = list(experiment_descriptor.keys())
    values = [experiment_descriptor[key] for key in keys]

    combinations = list(product(*values))

    # Create a dictionary with indexed combinations
    indexed_combinations = {i + 1: dict(zip(keys, combo)) for i, combo in enumerate(combinations)}


    # Save the indexed combinations to a JSON file
    with open(output, 'w') as outfile:
        json.dump(indexed_combinations, outfile, indent=4)

    print(f"Indexed combinations saved to {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read experiment descriptor and generate combinations.")
    parser.add_argument("--file", type=str, default="./metadata/experiment-descriptor.json",
                        help="Path to the CSV file containing stock symbols")
    parser.add_argument("--output", type=str, default="./metadata/experiment-index.json",
                        help="Path to the CSV file containing stock symbols")
    args = parser.parse_args()
    main(args.file, args.output)
