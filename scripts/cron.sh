#!/bin/bash
echo "I am running at $(date)"
cd /Users/jmordetsky/machine_learning_finance
source venv/bin/activate
python3 ./scripts/monitor.py
