python scripts/train-test-comparison.py --model-token advanced_lstm &&
python scripts/train-test-comparison.py --model-token transformer &&
python scripts/train-test-comparison.py --model-token lstm &&
python scripts/train-test-comparison.py --model-token advanced_lstm --data-type direction &&
python scripts/train-test-comparison.py --model-token transformer --data-type direction &&
python scripts/train-test-comparison.py --model-token lstm --data-type direction