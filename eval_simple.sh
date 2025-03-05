python scripts/experiments/easy_trainer.py --tickers ./metadata/basic-tickers.json --model vanilla.medium.extractors.layered.tanh --size 4096 --loss huber-sign --file-post-fix=_quarterly_magnitude --label PM --patience 1000 --seq-len=12 &&
python scripts/experiments/easy_trainer.py --category=easy_seq_24 --tickers ./metadata/basic-tickers.json --model vanilla.medium.extractors.layered.tanh --size 4096 --loss huber-sign --file-post-fix=_quarterly_magnitude --label PM --patience 1000 --seq-len=24 &&
python scripts/experiments/easy_evaler.py --tickers ./metadata/basic-tickers.json --model vanilla.medium.extractors.tanh --size 4096 --loss huber-sign --file-post-fix=_quarterly_magnitude --label PM  --seq-len=12 &&
python scripts/experiments/easy_evaler.py --category=easy_seq_24 --tickers ./metadata/basic-tickers.json --model vanilla.medium.extractors.tanh --size 4096 --loss huber-sign --file-post-fix=_quarterly_magnitude --label PM --seq-len=24



