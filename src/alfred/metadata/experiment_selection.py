import json
from alfred.utils import MongoConnectionStrings

connection = MongoConnectionStrings()


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
    def __init__(self, index_file, mongo=None, db=None):
        # Load JSON data from the provided file
        with open(index_file, 'r') as file:
            self.experiments = json.load(file)
        if mongo and db:
            self.mongo_client = connection.get_mongo_client()
            _db = self.mongo_client[db]
            self.collection = _db['runs']

    def get_current_state(self, namespace, build_descriptor_key):
        if self.mongo_client is None:
            raise Exception('MongoClient is not initialized. Supply mongo params to constructor')
        # Query to retrieve all 'COMPLETED' experiments
        cursor = self.collection.find({
            'status': {'$in': ['COMPLETED', 'RUNNING']},
            'experiment.name': namespace  # Filter by experiment name
        },
        {
            '_id': 1,  # Experiment ID
            'experiment.name': 1,  # Experiment name
            'config': 1,  # Configuration parameters
        })

        completed = {}

        # Iterate through the cursor to flatten and extract fields
        for doc in cursor:
            config = doc.get('config', {})
            key = build_descriptor_key(config)
            completed[key] = True
        return completed

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
