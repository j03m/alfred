#!/bin/bash
echo "I am running at $(date)"
export MY_PROXY="proxy.bloomberg.com"
export MY_PORT="81"
cd /Users/jmordetsky/machine_learning_finance
source venv/bin/activate
python3 ./scripts/monitor.py
