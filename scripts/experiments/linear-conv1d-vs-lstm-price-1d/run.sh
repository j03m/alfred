python scripts/train-test-comparison.py --model-token linear --epochs 1000 --predict-type price &&
python scripts/train-test-comparison.py --model-token linear-conv1d --epochs 1000 --predict-type price &&
python scripts/train-test-comparison.py --model-token lstm --epochs 1000 --predict-type price &&