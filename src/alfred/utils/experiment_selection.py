import json
def parse_ranges(ranges_str):
    ranges = []
    if not ranges_str:
        return ranges
    for part in ranges_str.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            ranges.extend(range(start, end + 1))
        else:
            ranges.append(int(part))
    return ranges

class ExperimentSelector:
    def __init__(self, index_file):
        # Load JSON data from the provided file
        with open(index_file, 'r') as file:
            self.experiments = json.load(file)

    def get(self, include_ranges, exclude_ranges):
        # Convert range strings to lists of indices
        include_list = parse_ranges(include_ranges)
        exclude_list = parse_ranges(exclude_ranges)

        # If no include range is specified, include all experiments by default
        if not include_list:
            include_list = list(map(int, self.experiments.keys()))

        # Remove excluded experiments from the include list
        include_list = [i for i in include_list if i not in exclude_list]

        # Run selected experiments
        selected_experiments = []
        for i in include_list:
            experiment = self.experiments.get(str(i))
            selected_experiments.append(experiment)

        return selected_experiments