#!/bin/bash
echo "I am running at $(date)"
source venv/bin/activate
python3 ./scripts/cache_data.py -f ./lists/training_list.csv
python3 ./scripts/cache_data.py -f ./lists/eval_list.csv
python3 ./scripts/calculate-analytics.py --train-set ./lists/training_list.csv
python3 ./scripts/calculate-analytics.py --train-set ./lists/eval_list.csv