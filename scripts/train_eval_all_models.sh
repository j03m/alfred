python scripts/train-test-comparison.py --model-token advanced_lstm &&
python scripts/train-test-comparison.py --model-token transformer &&
python scripts/train-test-comparison.py --model-token lstm &&
python scripts/train-test-comparison.py --model-token advanced_lstm --predict-type direction &&
python scripts/train-test-comparison.py --model-token transformer --predict-type direction &&
python scripts/train-test-comparison.py --model-token lstm --predict-type direction