(venv) jmordetsky in ~/alfred (main) > python scripts/experiments/easy_experiments.py --batch-size 256
Using device: mps
Using device: mps
Using device: mps
Primary connection failed: No servers found yet, Timeout: 3.0s, Topology Description: <TopologyDescription id: 67e05348968a448fb64cfc63, topology_type: Unknown, servers: [<ServerDescription ('192.168.156.236', 27017) server_type: Unknown, rtt: None>]>
Connected to MongoDB (fallback to localhost).
Using device: mps
Using device: mps
Using device: mps
WARNING - root - Changed type of config entry "model_name" from tuple to str
WARNING - root - Changed type of config entry "size" from tuple to int
INFO - alfred-experiments-2 - Running command 'run_experiment'
INFO - alfred-experiments-2 - Started run with ID "43"
Starting easy trainer
Prepping data
loading df 5100 of 5100loading model from config or creating model
Looking for: easy_experiments_lstm.medium.extractors.tanh_1024_12_1_0x69d76945_mps
Found model version 51 for easy_experiments_lstm.medium.extractors.tanh_1024_12_1_0x69d76945_mps.
/Users/jmordetsky/alfred/src/alfred/model_persistence/model_creation_and_storage.py:209: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  return torch.load(buffer)
Found scaler version 51 for easy_experiments_lstm.medium.extractors.tanh_1024_12_1_0x69d76945_mps.
Starting training:
epoch: 0 training seq 6254 of 6255Epoch 0 (time: 101.77405309677124) patience 0/1000 -  loss: 0.6836584081824163 vs best loss: 0.12451329277828335
Stats:  {'mae': 0.11836026608943939, 'r2': 0.00043141841888427734, 'pearson_corr': 0.022869711741805077, 'sign_accuracy': 0.670616978011821}
last learning rate: [0.00023730468750000005]
Predictions: [-0.03781008 -0.06291307 -0.00216268 -0.04744218 -0.04611241 -0.01525578
  0.00071909 -0.02439431 -0.07540724 -0.0214231  -0.03199005  0.02512889
 -0.02302256 -0.17189594 -0.14262356 -0.0414834  -0.04259878 -0.04148343
 -0.06228825 -0.01312777 -0.02543949  0.01747751  0.03587596 -0.06092349
  0.01504692]
Labels:      [-0.08787274 -0.09152545 -0.04292854 -0.08784047  0.03533891 -0.05643566
 -0.05928782  0.03962471 -0.09807798 -0.09987422 -0.04203699 -0.03463482
 -0.05065147 -0.17175965 -0.12384789  0.00029637  0.16428682  0.11106931
 -0.04769026  0.17175661  0.00659391 -0.05183325  0.08329966  0.04668785
  0.13037683]
<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=15.4 KiB (+15.4 KiB), count=268 (+268), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=12.2 KiB (+12.2 KiB), count=212 (+212), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/functional.py:77: size=3776 B (+3776 B), count=64 (+64), average=59 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=3127 B (+3127 B), count=53 (+53), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/bson/__init__.py:1173: size=1254 B (+1254 B), count=28 (+28), average=45 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torchmetrics/metric.py:768: size=1056 B (+1056 B), count=12 (+12), average=88 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:557: size=1024 B (+1024 B), count=8 (+8), average=128 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/pymongo/socket_checker.py:61: size=960 B (+960 B), count=6 (+6), average=160 B
epoch: 1 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=16.5 KiB (+16.5 KiB), count=286 (+286), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=10.5 KiB (+10.5 KiB), count=183 (+183), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=7704 B (+7704 B), count=107 (+107), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=7200 B (+7200 B), count=90 (+90), average=80 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/dataset.py:211: size=5376 B (+5376 B), count=96 (+96), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=5136 B (+5136 B), count=107 (+107), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/functional.py:77: size=4941 B (+4941 B), count=84 (+84), average=59 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=4307 B (+4307 B), count=73 (+73), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
epoch: 2 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=17.3 KiB (+17.3 KiB), count=300 (+300), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=12.7 KiB (+12.7 KiB), count=221 (+221), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=9576 B (+9576 B), count=133 (+133), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=9280 B (+9280 B), count=116 (+116), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=7888 B (+7888 B), count=140 (+140), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6384 B (+6384 B), count=133 (+133), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/functional.py:77: size=3891 B (+3891 B), count=66 (+66), average=59 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=3481 B (+3481 B), count=59 (+59), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
epoch: 3 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=17.5 KiB (+17.5 KiB), count=303 (+303), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=12.5 KiB (+12.5 KiB), count=217 (+217), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.1 KiB (+10.1 KiB), count=143 (+143), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10080 B (+10080 B), count=126 (+126), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=9288 B (+9288 B), count=165 (+165), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6864 B (+6864 B), count=143 (+143), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/functional.py:77: size=4242 B (+4242 B), count=72 (+72), average=59 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=3540 B (+3540 B), count=60 (+60), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
epoch: 4 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=18.1 KiB (+18.1 KiB), count=315 (+315), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=13.8 KiB (+13.8 KiB), count=239 (+239), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10080 B (+10080 B), count=126 (+126), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=9936 B (+9936 B), count=138 (+138), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=9736 B (+9736 B), count=173 (+173), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6624 B (+6624 B), count=138 (+138), average=48 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=3658 B (+3658 B), count=62 (+62), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
epoch: 5 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=19.0 KiB (+19.0 KiB), count=330 (+330), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=13.7 KiB (+13.7 KiB), count=350 (+350), average=40 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=13.1 KiB (+13.1 KiB), count=228 (+228), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.1 KiB (+10.1 KiB), count=143 (+143), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.0 KiB (+10.0 KiB), count=128 (+128), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=9440 B (+9440 B), count=168 (+168), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6864 B (+6864 B), count=143 (+143), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=4320 B (+4320 B), count=20 (+20), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=4160 B (+4160 B), count=20 (+20), average=208 B
epoch: 6 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=19.4 KiB (+19.4 KiB), count=336 (+336), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=14.0 KiB (+14.0 KiB), count=243 (+243), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.1 KiB (+10.1 KiB), count=144 (+144), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.0 KiB (+10.0 KiB), count=128 (+128), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=9792 B (+9792 B), count=174 (+174), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6912 B (+6912 B), count=144 (+144), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/functional.py:77: size=3658 B (+3658 B), count=62 (+62), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
epoch: 7 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=20.2 KiB (+20.2 KiB), count=350 (+350), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=14.5 KiB (+14.5 KiB), count=252 (+252), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10224 B (+10224 B), count=142 (+142), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10160 B (+10160 B), count=127 (+127), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=9296 B (+9296 B), count=166 (+166), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=8400 B (+8400 B), count=210 (+210), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6816 B (+6816 B), count=142 (+142), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/functional.py:77: size=3360 B (+3360 B), count=57 (+57), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
epoch: 8 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=20.7 KiB (+20.7 KiB), count=359 (+359), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.5 KiB (+16.5 KiB), count=287 (+287), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10000 B (+10000 B), count=125 (+125), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=9936 B (+9936 B), count=138 (+138), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=9464 B (+9464 B), count=169 (+169), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6624 B (+6624 B), count=138 (+138), average=48 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=3186 B (+3186 B), count=54 (+54), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/functional.py:77: size=2357 B (+2357 B), count=40 (+40), average=59 B
epoch: 9 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=21.1 KiB (+21.1 KiB), count=366 (+366), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.2 KiB (+16.2 KiB), count=281 (+281), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10160 B (+10160 B), count=127 (+127), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=9288 B (+9288 B), count=129 (+129), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=8680 B (+8680 B), count=155 (+155), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6192 B (+6192 B), count=129 (+129), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 10 training seq 6254 of 6255Epoch 10 (time: 97.52890586853027) patience 9/1000 -  loss: 0.6042731129913498 vs best loss: 0.12451329277828335
Stats:  {'mae': 0.10827154666185379, 'r2': 0.003976404666900635, 'pearson_corr': 0.06423698365688324, 'sign_accuracy': 0.7090925442036842}
last learning rate: [0.00023730468750000005]
Predictions: [ 0.03597258 -0.03597998 -0.05623475  0.05498223  0.06687843 -0.07732941
 -0.08339223 -0.03074242 -0.05500175 -0.06859163  0.05827431 -0.2531875
 -0.03329365 -0.02472073 -0.03192283 -0.05863111  0.02214458  0.12314288
 -0.1401672  -0.10905682 -0.10151704 -0.01157212 -0.02779259 -0.07719371
  0.00790655]
Labels:      [-0.29100484 -0.07460213 -0.0182813   0.04710117 -0.00251251  0.00725342
  0.0015658  -0.17126627 -0.04601262 -0.14968888 -0.0999698  -0.31815138
 -0.06190857 -0.0667424  -0.08007572 -0.11794133  0.01971505  0.27448556
 -0.14492407 -0.08167478 -0.09959536 -0.08794419  0.04598282 -0.05245331
  0.00302857]
<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=20.7 KiB (+20.7 KiB), count=360 (+360), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.0 KiB (+16.0 KiB), count=277 (+277), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10080 B (+10080 B), count=126 (+126), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=9128 B (+9128 B), count=163 (+163), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:481: size=3264 B (+3264 B), count=16 (+16), average=204 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=3009 B (+3009 B), count=51 (+51), average=59 B
epoch: 11 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=22.0 KiB (+22.0 KiB), count=381 (+381), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=17.5 KiB (+17.5 KiB), count=304 (+304), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.2 KiB (+10.2 KiB), count=130 (+130), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10152 B (+10152 B), count=141 (+141), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=9296 B (+9296 B), count=166 (+166), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6768 B (+6768 B), count=141 (+141), average=48 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=2832 B (+2832 B), count=48 (+48), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/functional.py:77: size=1944 B (+1944 B), count=33 (+33), average=59 B
epoch: 12 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=22.2 KiB (+22.2 KiB), count=386 (+386), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.0 KiB (+16.0 KiB), count=277 (+277), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.3 KiB (+10.3 KiB), count=132 (+132), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=9216 B (+9216 B), count=128 (+128), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=9184 B (+9184 B), count=164 (+164), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6144 B (+6144 B), count=128 (+128), average=48 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=4012 B (+4012 B), count=68 (+68), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
epoch: 13 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=22.4 KiB (+22.4 KiB), count=389 (+389), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.0 KiB (+16.0 KiB), count=277 (+277), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=13.7 KiB (+13.7 KiB), count=350 (+350), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=134 (+134), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=9936 B (+9936 B), count=138 (+138), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=9520 B (+9520 B), count=170 (+170), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6624 B (+6624 B), count=138 (+138), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=4320 B (+4320 B), count=20 (+20), average=216 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=4307 B (+4307 B), count=73 (+73), average=59 B
epoch: 14 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=22.6 KiB (+22.6 KiB), count=392 (+392), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.7 KiB (+16.7 KiB), count=289 (+289), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.4 KiB (+10.4 KiB), count=133 (+133), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10224 B (+10224 B), count=142 (+142), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=10080 B (+10080 B), count=180 (+180), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6816 B (+6816 B), count=142 (+142), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 15 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=22.8 KiB (+22.8 KiB), count=395 (+395), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.8 KiB (+16.8 KiB), count=291 (+291), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.3 KiB (+10.3 KiB), count=132 (+132), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.2 KiB (+10.2 KiB), count=145 (+145), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=10080 B (+10080 B), count=180 (+180), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=8400 B (+8400 B), count=210 (+210), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6960 B (+6960 B), count=145 (+145), average=48 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=3835 B (+3835 B), count=65 (+65), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
epoch: 16 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.5 KiB (+23.5 KiB), count=408 (+408), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.9 KiB (+16.9 KiB), count=294 (+294), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.3 KiB (+10.3 KiB), count=132 (+132), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.2 KiB (+10.2 KiB), count=145 (+145), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=10192 B (+10192 B), count=182 (+182), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6960 B (+6960 B), count=145 (+145), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=2714 B (+2714 B), count=46 (+46), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/functional.py:77: size=2475 B (+2475 B), count=42 (+42), average=59 B
epoch: 17 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.3 KiB (+23.3 KiB), count=404 (+404), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.8 KiB (+16.8 KiB), count=292 (+292), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=134 (+134), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=9360 B (+9360 B), count=130 (+130), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=9352 B (+9352 B), count=167 (+167), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6240 B (+6240 B), count=130 (+130), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 18 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.3 KiB (+23.3 KiB), count=405 (+405), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=17.7 KiB (+17.7 KiB), count=307 (+307), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=13.7 KiB (+13.7 KiB), count=350 (+350), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.1 KiB (+10.1 KiB), count=144 (+144), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=9800 B (+9800 B), count=175 (+175), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6912 B (+6912 B), count=144 (+144), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=4320 B (+4320 B), count=20 (+20), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=4160 B (+4160 B), count=20 (+20), average=208 B
epoch: 19 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.4 KiB (+23.4 KiB), count=406 (+406), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=17.1 KiB (+17.1 KiB), count=297 (+297), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=135 (+135), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.1 KiB (+10.1 KiB), count=144 (+144), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=10.0 KiB (+10.0 KiB), count=183 (+183), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6912 B (+6912 B), count=144 (+144), average=48 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=3599 B (+3599 B), count=61 (+61), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
epoch: 20 training seq 6254 of 6255Epoch 20 (time: 97.67028880119324) patience 19/1000 -  loss: 0.5568100178985002 vs best loss: 0.12451329277828335
Stats:  {'mae': 0.10044027119874954, 'r2': 0.007390201091766357, 'pearson_corr': 0.08758827298879623, 'sign_accuracy': 0.7317002493092783}
last learning rate: [0.00023730468750000005]
Predictions: [-0.02599984 -0.02623405 -0.07702574 -0.03493683 -0.0658225  -0.03081553
  0.03821781 -0.04465759 -0.02184262 -0.10562453 -0.00657034 -0.04937831
 -0.09482093 -0.00671566  0.35096836 -0.08930335 -0.02992487 -0.00875568
 -0.09153177 -0.05266547 -0.060371   -0.21468897  0.00436431 -0.17175266
 -0.01101094]
Labels:      [ 0.3317087   0.08593825 -0.06479596 -0.12284987 -0.19343261  0.00903961
 -0.07838755  0.06714132  0.09250656 -0.0194583  -0.06095636  0.08769502
 -0.08068889 -0.05150487  0.05633585  0.00919388 -0.04130777  0.02007554
 -0.08695284 -0.11410049 -0.20381258 -0.40816104  0.04756253 -0.19923232
 -0.03892058]
<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=22.4 KiB (+22.4 KiB), count=388 (+388), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.9 KiB (+16.9 KiB), count=293 (+293), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=134 (+134), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=10.1 KiB (+10.1 KiB), count=184 (+184), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=8400 B (+8400 B), count=210 (+210), average=40 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=2714 B (+2714 B), count=46 (+46), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=2592 B (+2592 B), count=12 (+12), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=2496 B (+2496 B), count=12 (+12), average=208 B
epoch: 21 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=409 (+409), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=17.6 KiB (+17.6 KiB), count=306 (+306), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.4 KiB (+10.4 KiB), count=133 (+133), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=10136 B (+10136 B), count=181 (+181), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=9720 B (+9720 B), count=135 (+135), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6480 B (+6480 B), count=135 (+135), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=2800 B (+2800 B), count=70 (+70), average=40 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=2596 B (+2596 B), count=44 (+44), average=59 B
epoch: 22 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.5 KiB (+23.5 KiB), count=408 (+408), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=17.5 KiB (+17.5 KiB), count=303 (+303), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=134 (+134), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=10192 B (+10192 B), count=182 (+182), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=9936 B (+9936 B), count=138 (+138), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6624 B (+6624 B), count=138 (+138), average=48 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=3186 B (+3186 B), count=54 (+54), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/functional.py:77: size=1826 B (+1826 B), count=31 (+31), average=59 B
epoch: 23 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.3 KiB (+23.3 KiB), count=404 (+404), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=17.5 KiB (+17.5 KiB), count=304 (+304), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=134 (+134), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=9744 B (+9744 B), count=174 (+174), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=9288 B (+9288 B), count=129 (+129), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6192 B (+6192 B), count=129 (+129), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 24 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=409 (+409), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.0 KiB (+16.0 KiB), count=277 (+277), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.1 KiB (+10.1 KiB), count=143 (+143), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=10136 B (+10136 B), count=181 (+181), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6864 B (+6864 B), count=143 (+143), average=48 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=4071 B (+4071 B), count=69 (+69), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
epoch: 25 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=410 (+410), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=17.3 KiB (+17.3 KiB), count=301 (+301), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=135 (+135), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=10.2 KiB (+10.2 KiB), count=186 (+186), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10008 B (+10008 B), count=139 (+139), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6672 B (+6672 B), count=139 (+139), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 26 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=410 (+410), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=17.8 KiB (+17.8 KiB), count=309 (+309), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=135 (+135), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=10.4 KiB (+10.4 KiB), count=190 (+190), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10224 B (+10224 B), count=142 (+142), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6816 B (+6816 B), count=142 (+142), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 27 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.5 KiB (+23.5 KiB), count=408 (+408), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=17.1 KiB (+17.1 KiB), count=296 (+296), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=134 (+134), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=10.4 KiB (+10.4 KiB), count=190 (+190), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10008 B (+10008 B), count=139 (+139), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6672 B (+6672 B), count=139 (+139), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 28 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.7 KiB (+23.7 KiB), count=412 (+412), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.8 KiB (+16.8 KiB), count=291 (+291), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=10.7 KiB (+10.7 KiB), count=195 (+195), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=135 (+135), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10224 B (+10224 B), count=142 (+142), average=72 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=8400 B (+8400 B), count=210 (+210), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6816 B (+6816 B), count=142 (+142), average=48 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=2891 B (+2891 B), count=49 (+49), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
epoch: 29 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.9 KiB (+23.9 KiB), count=414 (+414), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.7 KiB (+16.7 KiB), count=290 (+290), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=10.6 KiB (+10.6 KiB), count=194 (+194), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=135 (+135), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10152 B (+10152 B), count=141 (+141), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6768 B (+6768 B), count=141 (+141), average=48 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=3540 B (+3540 B), count=60 (+60), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/functional.py:77: size=1947 B (+1947 B), count=33 (+33), average=59 B
epoch: 30 training seq 6254 of 6255Epoch 30 (time: 97.55515003204346) patience 29/1000 -  loss: 0.49539129058649906 vs best loss: 0.12451329277828335
Stats:  {'mae': 0.09174151718616486, 'r2': 0.00952214002609253, 'pearson_corr': 0.09918589890003204, 'sign_accuracy': 0.7615624047603583}
last learning rate: [0.00023730468750000005]
Predictions: [-0.05998076  0.05052208 -0.0603047  -0.04470471  0.25208798 -0.06031689
 -0.04840583 -0.09894987 -0.09492614 -0.00030581 -0.16327433 -0.14381583
  0.00453108 -0.08167928 -0.07805041 -0.06311584  0.12392846  0.12819456
 -0.2309925  -0.06267574 -0.15687671 -0.08494347 -0.07883368 -0.08761911
  0.26379964]
Labels:      [-8.5453697e-02  1.2451555e+00 -5.6330465e-02  2.5044331e-02
  2.1767083e-01 -4.2759672e-02 -2.0066480e-01  2.0724152e-01
 -1.8540828e-01 -1.5171580e-01 -2.0386742e-01 -1.8965589e-01
  8.5240342e-03 -5.5751350e-02  1.0228597e-03  4.4071604e-02
  7.3315218e-02  2.1948466e-02 -1.0538655e-01 -9.3858153e-02
 -2.0077738e-01  7.3462635e-02 -5.2453309e-02 -3.5407808e-02
  1.7570525e-01]
<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=22.7 KiB (+22.7 KiB), count=394 (+394), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=17.6 KiB (+17.6 KiB), count=306 (+306), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=10.5 KiB (+10.5 KiB), count=192 (+192), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=134 (+134), average=80 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:481: size=3264 B (+3264 B), count=16 (+16), average=204 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
epoch: 31 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.3 KiB (+23.3 KiB), count=404 (+404), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.4 KiB (+16.4 KiB), count=284 (+284), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=10.6 KiB (+10.6 KiB), count=193 (+193), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=134 (+134), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10008 B (+10008 B), count=139 (+139), average=72 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=8400 B (+8400 B), count=210 (+210), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6672 B (+6672 B), count=139 (+139), average=48 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=3245 B (+3245 B), count=55 (+55), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
epoch: 32 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.5 KiB (+23.5 KiB), count=407 (+407), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.5 KiB (+16.5 KiB), count=287 (+287), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=11.0 KiB (+11.0 KiB), count=201 (+201), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.7 KiB (+10.7 KiB), count=152 (+152), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=135 (+135), average=80 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=8400 B (+8400 B), count=210 (+210), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=7296 B (+7296 B), count=152 (+152), average=48 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=2891 B (+2891 B), count=49 (+49), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
epoch: 33 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.5 KiB (+23.5 KiB), count=407 (+407), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.1 KiB (+16.1 KiB), count=279 (+279), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=11.0 KiB (+11.0 KiB), count=202 (+202), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.3 KiB (+10.3 KiB), count=146 (+146), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=7008 B (+7008 B), count=146 (+146), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 34 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.7 KiB (+23.7 KiB), count=411 (+411), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.7 KiB (+16.7 KiB), count=289 (+289), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=11.1 KiB (+11.1 KiB), count=203 (+203), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=135 (+135), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.4 KiB (+10.4 KiB), count=148 (+148), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=7104 B (+7104 B), count=148 (+148), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 35 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.5 KiB (+23.5 KiB), count=408 (+408), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.2 KiB (+16.2 KiB), count=282 (+282), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=11.3 KiB (+11.3 KiB), count=207 (+207), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=135 (+135), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.5 KiB (+10.5 KiB), count=149 (+149), average=72 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=8400 B (+8400 B), count=210 (+210), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=7152 B (+7152 B), count=149 (+149), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/functional.py:77: size=3183 B (+3183 B), count=54 (+54), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
epoch: 36 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.7 KiB (+23.7 KiB), count=412 (+412), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=17.0 KiB (+17.0 KiB), count=295 (+295), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=11.0 KiB (+11.0 KiB), count=201 (+201), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.3 KiB (+10.3 KiB), count=132 (+132), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.1 KiB (+10.1 KiB), count=144 (+144), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6912 B (+6912 B), count=144 (+144), average=48 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=3068 B (+3068 B), count=52 (+52), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/functional.py:77: size=2183 B (+2183 B), count=37 (+37), average=59 B
epoch: 37 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=410 (+410), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.4 KiB (+16.4 KiB), count=284 (+284), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=10.7 KiB (+10.7 KiB), count=196 (+196), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=135 (+135), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=9936 B (+9936 B), count=138 (+138), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6624 B (+6624 B), count=138 (+138), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 38 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.5 KiB (+23.5 KiB), count=408 (+408), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.2 KiB (+16.2 KiB), count=281 (+281), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=13.7 KiB (+13.7 KiB), count=350 (+350), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=11.1 KiB (+11.1 KiB), count=203 (+203), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.6 KiB (+10.6 KiB), count=151 (+151), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=7248 B (+7248 B), count=151 (+151), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=4320 B (+4320 B), count=20 (+20), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=4160 B (+4160 B), count=20 (+20), average=208 B
epoch: 39 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.8 KiB (+23.8 KiB), count=413 (+413), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.5 KiB (+16.5 KiB), count=286 (+286), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=11.4 KiB (+11.4 KiB), count=209 (+209), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.6 KiB (+10.6 KiB), count=151 (+151), average=72 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=8400 B (+8400 B), count=210 (+210), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=7248 B (+7248 B), count=151 (+151), average=48 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=3363 B (+3363 B), count=57 (+57), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
epoch: 40 training seq 6254 of 6255Epoch 40 (time: 97.41945695877075) patience 39/1000 -  loss: 0.4406086672141182 vs best loss: 0.12451329277828335
Stats:  {'mae': 0.08438727259635925, 'r2': 0.011123418807983398, 'pearson_corr': 0.10720381885766983, 'sign_accuracy': 0.7883868840337143}
last learning rate: [0.00023730468750000005]
Predictions: [ 0.2857232   0.05077306 -0.2708456  -0.08004629 -0.06075302 -0.22994283
 -0.14579909 -0.05766548 -0.13968964 -0.00886329  0.05923666 -0.03834342
 -0.02916846 -0.35937127 -0.00565002  0.06560533 -0.11175716  0.01350714
 -0.00837002 -0.04156749  0.05606551 -0.00456969  0.08566252  0.016467
  0.09687027]
Labels:      [ 0.16292736 -0.00759479 -0.21458057 -0.18944706 -0.28863406 -0.35818326
 -0.18715012 -0.10135836 -0.16308622 -0.00298625  0.07005274 -0.05825843
  0.00858614 -0.2845773  -0.03918486  0.00580419 -0.16617128  0.02841873
 -0.07732452 -0.02824682  0.07194313  0.00995606  0.08978129  0.0419392
  0.02605252]
<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=22.9 KiB (+22.9 KiB), count=397 (+397), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.2 KiB (+16.2 KiB), count=281 (+281), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=11.5 KiB (+11.5 KiB), count=210 (+210), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:481: size=3264 B (+3264 B), count=16 (+16), average=204 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
epoch: 41 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=410 (+410), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.9 KiB (+16.9 KiB), count=293 (+293), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=11.2 KiB (+11.2 KiB), count=205 (+205), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=134 (+134), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=9792 B (+9792 B), count=136 (+136), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6528 B (+6528 B), count=136 (+136), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=2800 B (+2800 B), count=70 (+70), average=40 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=2655 B (+2655 B), count=45 (+45), average=59 B
epoch: 42 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.7 KiB (+23.7 KiB), count=412 (+412), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=17.5 KiB (+17.5 KiB), count=304 (+304), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=11.4 KiB (+11.4 KiB), count=208 (+208), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.4 KiB (+10.4 KiB), count=148 (+148), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=7104 B (+7104 B), count=148 (+148), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=2478 B (+2478 B), count=42 (+42), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/functional.py:77: size=2121 B (+2121 B), count=36 (+36), average=59 B
epoch: 43 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.7 KiB (+23.7 KiB), count=411 (+411), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.6 KiB (+16.6 KiB), count=288 (+288), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=11.5 KiB (+11.5 KiB), count=210 (+210), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=135 (+135), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10224 B (+10224 B), count=142 (+142), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6816 B (+6816 B), count=142 (+142), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 44 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=409 (+409), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.1 KiB (+16.1 KiB), count=279 (+279), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=11.8 KiB (+11.8 KiB), count=216 (+216), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.8 KiB (+10.8 KiB), count=138 (+138), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.5 KiB (+10.5 KiB), count=149 (+149), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=7152 B (+7152 B), count=149 (+149), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 45 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.5 KiB (+23.5 KiB), count=408 (+408), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.2 KiB (+16.2 KiB), count=281 (+281), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=12.0 KiB (+12.0 KiB), count=220 (+220), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=135 (+135), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.2 KiB (+10.2 KiB), count=145 (+145), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6960 B (+6960 B), count=145 (+145), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 46 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=409 (+409), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=15.3 KiB (+15.3 KiB), count=265 (+265), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=12.3 KiB (+12.3 KiB), count=315 (+315), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=11.8 KiB (+11.8 KiB), count=215 (+215), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.7 KiB (+10.7 KiB), count=137 (+137), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.1 KiB (+10.1 KiB), count=143 (+143), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6864 B (+6864 B), count=143 (+143), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3888 B (+3888 B), count=18 (+18), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3744 B (+3744 B), count=18 (+18), average=208 B
epoch: 47 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.7 KiB (+23.7 KiB), count=411 (+411), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.0 KiB (+16.0 KiB), count=277 (+277), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=12.0 KiB (+12.0 KiB), count=219 (+219), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=135 (+135), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.2 KiB (+10.2 KiB), count=145 (+145), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6960 B (+6960 B), count=145 (+145), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 48 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=409 (+409), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=17.2 KiB (+17.2 KiB), count=299 (+299), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=12.0 KiB (+12.0 KiB), count=220 (+220), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=135 (+135), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.1 KiB (+10.1 KiB), count=144 (+144), average=72 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=8400 B (+8400 B), count=210 (+210), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6912 B (+6912 B), count=144 (+144), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=2592 B (+2592 B), count=12 (+12), average=216 B
epoch: 49 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.7 KiB (+23.7 KiB), count=411 (+411), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=17.1 KiB (+17.1 KiB), count=297 (+297), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=12.0 KiB (+12.0 KiB), count=219 (+219), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.3 KiB (+10.3 KiB), count=132 (+132), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.1 KiB (+10.1 KiB), count=144 (+144), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6912 B (+6912 B), count=144 (+144), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=2714 B (+2714 B), count=46 (+46), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/functional.py:77: size=2593 B (+2593 B), count=44 (+44), average=59 B
epoch: 50 training seq 6254 of 6255Epoch 50 (time: 97.5314610004425) patience 49/1000 -  loss: 0.39511590568712807 vs best loss: 0.12451329277828335
Stats:  {'mae': 0.07830123603343964, 'r2': 0.011777400970458984, 'pearson_corr': 0.10987775027751923, 'sign_accuracy': 0.8106748337521796}
last learning rate: [0.00023730468750000005]
Predictions: [ 0.45752886 -0.27343792 -0.02641609  0.01127784 -0.04684318  0.06148107
  0.00054288 -0.03280834 -0.12415427 -0.05118195 -0.18110019 -0.09022445
 -0.06476793 -0.05467383  0.04069331  0.02341453 -0.14848037 -0.10357658
 -0.12872541  0.04080371 -0.08989034 -0.10945906  0.05719044 -0.06244102
  0.0302865 ]
Labels:      [ 0.47971138 -0.27506816 -0.04570054 -0.09092772 -0.04542266  0.10127082
  0.01790367 -0.03733501 -0.11155459  0.0323145  -0.22912514 -0.06044714
 -0.0737609  -0.07248969  0.03559222 -0.15519863 -0.16187973  0.03091298
 -0.10923849  0.06538694 -0.07099386 -0.04232888  0.04154205 -0.05245331
  0.01250122]
<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=22.9 KiB (+22.9 KiB), count=397 (+397), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.4 KiB (+16.4 KiB), count=285 (+285), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=11.5 KiB (+11.5 KiB), count=210 (+210), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=135 (+135), average=80 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:481: size=3264 B (+3264 B), count=16 (+16), average=204 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=2832 B (+2832 B), count=48 (+48), average=59 B
epoch: 51 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=409 (+409), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.2 KiB (+16.2 KiB), count=282 (+282), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=11.8 KiB (+11.8 KiB), count=215 (+215), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=9648 B (+9648 B), count=134 (+134), average=72 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=8400 B (+8400 B), count=210 (+210), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6432 B (+6432 B), count=134 (+134), average=48 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=3304 B (+3304 B), count=56 (+56), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
epoch: 52 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=410 (+410), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.1 KiB (+16.1 KiB), count=279 (+279), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=12.1 KiB (+12.1 KiB), count=221 (+221), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=135 (+135), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.1 KiB (+10.1 KiB), count=144 (+144), average=72 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=8400 B (+8400 B), count=210 (+210), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6912 B (+6912 B), count=144 (+144), average=48 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=3422 B (+3422 B), count=58 (+58), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/functional.py:77: size=3124 B (+3124 B), count=53 (+53), average=59 B
epoch: 53 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.7 KiB (+23.7 KiB), count=412 (+412), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.5 KiB (+16.5 KiB), count=287 (+287), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=12.4 KiB (+12.4 KiB), count=226 (+226), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.7 KiB (+10.7 KiB), count=137 (+137), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.3 KiB (+10.3 KiB), count=147 (+147), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=7056 B (+7056 B), count=147 (+147), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=3363 B (+3363 B), count=57 (+57), average=59 B
epoch: 54 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=409 (+409), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.2 KiB (+16.2 KiB), count=281 (+281), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=12.4 KiB (+12.4 KiB), count=227 (+227), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10152 B (+10152 B), count=141 (+141), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6768 B (+6768 B), count=141 (+141), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 55 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=409 (+409), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.2 KiB (+16.2 KiB), count=282 (+282), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=12.6 KiB (+12.6 KiB), count=230 (+230), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.3 KiB (+10.3 KiB), count=146 (+146), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=7008 B (+7008 B), count=146 (+146), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 56 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=410 (+410), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.7 KiB (+16.7 KiB), count=289 (+289), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=12.5 KiB (+12.5 KiB), count=229 (+229), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=135 (+135), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.1 KiB (+10.1 KiB), count=144 (+144), average=72 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=8400 B (+8400 B), count=210 (+210), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6912 B (+6912 B), count=144 (+144), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=2714 B (+2714 B), count=46 (+46), average=59 B
epoch: 57 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=410 (+410), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=15.8 KiB (+15.8 KiB), count=275 (+275), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=12.9 KiB (+12.9 KiB), count=235 (+235), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=135 (+135), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.3 KiB (+10.3 KiB), count=147 (+147), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=7056 B (+7056 B), count=147 (+147), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 58 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.7 KiB (+23.7 KiB), count=411 (+411), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=15.6 KiB (+15.6 KiB), count=270 (+270), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=12.7 KiB (+12.7 KiB), count=233 (+233), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=135 (+135), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10224 B (+10224 B), count=142 (+142), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6816 B (+6816 B), count=142 (+142), average=48 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=3717 B (+3717 B), count=63 (+63), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
epoch: 59 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.5 KiB (+23.5 KiB), count=408 (+408), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.1 KiB (+16.1 KiB), count=279 (+279), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=12.9 KiB (+12.9 KiB), count=235 (+235), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=135 (+135), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.3 KiB (+10.3 KiB), count=146 (+146), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=7008 B (+7008 B), count=146 (+146), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=3363 B (+3363 B), count=57 (+57), average=59 B
epoch: 60 training seq 6254 of 6255Epoch 60 (time: 97.48482894897461) patience 59/1000 -  loss: 0.36753275971509797 vs best loss: 0.12451329277828335
Stats:  {'mae': 0.07477139681577682, 'r2': 0.012277483940124512, 'pearson_corr': 0.11217671632766724, 'sign_accuracy': 0.8242494491713839}
last learning rate: [0.00023730468750000005]
Predictions: [-0.05025473 -0.05374118 -0.06626771 -0.07946236 -0.09069422 -0.13932726
 -0.16258352  0.5156126  -0.06198008 -0.01834163  0.1865361  -0.06072897
 -0.10877251 -0.02754209 -0.134292    0.04962555 -0.06428836  0.00141575
 -0.04751742 -0.08080799 -0.00331829 -0.12275165 -0.06525882 -0.21853448
 -0.00181048]
Labels:      [-0.06520498 -0.08961862  0.11462782 -0.10343463 -0.09469748 -0.18488063
 -0.18944709  0.3355613  -0.03115692  0.00700824  0.21951954 -0.01678981
 -0.09143889  0.04060253 -0.23823088  0.13646545  0.02157436  0.02373161
 -0.00168366 -0.04018545  0.1187448  -0.08975732 -0.10736232 -0.19093794
 -0.0181277 ]
<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=22.9 KiB (+22.9 KiB), count=397 (+397), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=15.9 KiB (+15.9 KiB), count=276 (+276), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=13.0 KiB (+13.0 KiB), count=237 (+237), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=8400 B (+8400 B), count=210 (+210), average=40 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=3009 B (+3009 B), count=51 (+51), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/functional.py:77: size=2891 B (+2891 B), count=49 (+49), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=2592 B (+2592 B), count=12 (+12), average=216 B
epoch: 61 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.5 KiB (+23.5 KiB), count=408 (+408), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.3 KiB (+16.3 KiB), count=283 (+283), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=12.5 KiB (+12.5 KiB), count=229 (+229), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10152 B (+10152 B), count=141 (+141), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6768 B (+6768 B), count=141 (+141), average=48 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=3304 B (+3304 B), count=56 (+56), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=2800 B (+2800 B), count=70 (+70), average=40 B
epoch: 62 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.7 KiB (+23.7 KiB), count=411 (+411), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.0 KiB (+16.0 KiB), count=277 (+277), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=12.7 KiB (+12.7 KiB), count=233 (+233), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10224 B (+10224 B), count=142 (+142), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6816 B (+6816 B), count=142 (+142), average=48 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=3068 B (+3068 B), count=52 (+52), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=2800 B (+2800 B), count=70 (+70), average=40 B
epoch: 63 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.7 KiB (+23.7 KiB), count=412 (+412), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.0 KiB (+16.0 KiB), count=278 (+278), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=13.7 KiB (+13.7 KiB), count=350 (+350), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=12.8 KiB (+12.8 KiB), count=234 (+234), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=135 (+135), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10152 B (+10152 B), count=141 (+141), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6768 B (+6768 B), count=141 (+141), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=4320 B (+4320 B), count=20 (+20), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=4160 B (+4160 B), count=20 (+20), average=208 B
epoch: 64 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=410 (+410), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.5 KiB (+16.5 KiB), count=286 (+286), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=13.7 KiB (+13.7 KiB), count=350 (+350), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=12.9 KiB (+12.9 KiB), count=236 (+236), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.3 KiB (+10.3 KiB), count=146 (+146), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=7008 B (+7008 B), count=146 (+146), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=4320 B (+4320 B), count=20 (+20), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=4160 B (+4160 B), count=20 (+20), average=208 B
epoch: 65 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.7 KiB (+23.7 KiB), count=411 (+411), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.7 KiB (+16.7 KiB), count=290 (+290), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=12.8 KiB (+12.8 KiB), count=234 (+234), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=135 (+135), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.1 KiB (+10.1 KiB), count=144 (+144), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6912 B (+6912 B), count=144 (+144), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 66 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.7 KiB (+23.7 KiB), count=411 (+411), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.5 KiB (+16.5 KiB), count=286 (+286), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=13.0 KiB (+13.0 KiB), count=238 (+238), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=134 (+134), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.2 KiB (+10.2 KiB), count=145 (+145), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6960 B (+6960 B), count=145 (+145), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 67 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.7 KiB (+23.7 KiB), count=412 (+412), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.4 KiB (+16.4 KiB), count=285 (+285), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=13.0 KiB (+13.0 KiB), count=237 (+237), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=135 (+135), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10224 B (+10224 B), count=142 (+142), average=72 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=8400 B (+8400 B), count=210 (+210), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6816 B (+6816 B), count=142 (+142), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/functional.py:77: size=2891 B (+2891 B), count=49 (+49), average=59 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=2891 B (+2891 B), count=49 (+49), average=59 B
epoch: 68 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.8 KiB (+23.8 KiB), count=413 (+413), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.4 KiB (+16.4 KiB), count=285 (+285), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=13.2 KiB (+13.2 KiB), count=242 (+242), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=12.3 KiB (+12.3 KiB), count=315 (+315), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.7 KiB (+10.7 KiB), count=137 (+137), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.1 KiB (+10.1 KiB), count=144 (+144), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6912 B (+6912 B), count=144 (+144), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3888 B (+3888 B), count=18 (+18), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3744 B (+3744 B), count=18 (+18), average=208 B
epoch: 69 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=410 (+410), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.9 KiB (+16.9 KiB), count=294 (+294), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=12.9 KiB (+12.9 KiB), count=235 (+235), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=135 (+135), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10008 B (+10008 B), count=139 (+139), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6672 B (+6672 B), count=139 (+139), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 70 training seq 6254 of 6255Epoch 70 (time: 97.67838311195374) patience 69/1000 -  loss: 0.35148029195019703 vs best loss: 0.12451329277828335
Stats:  {'mae': 0.07275569438934326, 'r2': 0.01283407211303711, 'pearson_corr': 0.11479611694812775, 'sign_accuracy': 0.8321683911807467}
last learning rate: [0.00023730468750000005]
Predictions: [ 0.06701578 -0.03459392 -0.00768666 -0.07431993  0.029527   -0.00793569
 -0.08302391 -0.10800704  0.11442494  0.00936337 -0.04972552  0.05119301
  0.10071073  0.04349431  0.05532901 -0.25357884  0.0448671  -0.19359122
 -0.05571634 -0.200287   -0.10219156  0.20798768 -0.46834293 -0.02015999
  0.05811232]
Labels:      [-0.02438908 -0.07777123 -0.05484518 -0.14094925 -0.05245331 -0.01696141
 -0.17215343 -0.14344858  0.15309942 -0.02758712 -0.12306489  0.04070242
  0.12562732  0.14285761  0.02400179 -0.25794393 -0.13153777 -0.20766442
 -0.05818126 -0.17446332 -0.18048978  0.052798   -0.5873564  -0.0589085
  0.06317352]
<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.0 KiB (+23.0 KiB), count=400 (+400), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.2 KiB (+16.2 KiB), count=282 (+282), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=13.2 KiB (+13.2 KiB), count=242 (+242), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=12.3 KiB (+12.3 KiB), count=315 (+315), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.9 KiB (+10.9 KiB), count=139 (+139), average=80 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3888 B (+3888 B), count=18 (+18), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3744 B (+3744 B), count=18 (+18), average=208 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:481: size=3672 B (+3672 B), count=18 (+18), average=204 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=2832 B (+2832 B), count=48 (+48), average=59 B
epoch: 71 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=409 (+409), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.3 KiB (+16.3 KiB), count=283 (+283), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=13.3 KiB (+13.3 KiB), count=243 (+243), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=134 (+134), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.1 KiB (+10.1 KiB), count=144 (+144), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6912 B (+6912 B), count=144 (+144), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/functional.py:77: size=2711 B (+2711 B), count=46 (+46), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/nn/modules/rnn.py:917: size=2596 B (+2596 B), count=44 (+44), average=59 B
epoch: 72 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=410 (+410), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.6 KiB (+16.6 KiB), count=288 (+288), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=13.1 KiB (+13.1 KiB), count=239 (+239), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.7 KiB (+10.7 KiB), count=137 (+137), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=9864 B (+9864 B), count=137 (+137), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6576 B (+6576 B), count=137 (+137), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 73 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.5 KiB (+23.5 KiB), count=408 (+408), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=17.3 KiB (+17.3 KiB), count=300 (+300), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=13.7 KiB (+13.7 KiB), count=350 (+350), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=13.5 KiB (+13.5 KiB), count=247 (+247), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.7 KiB (+10.7 KiB), count=137 (+137), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.5 KiB (+10.5 KiB), count=149 (+149), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=7152 B (+7152 B), count=149 (+149), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=4320 B (+4320 B), count=20 (+20), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=4160 B (+4160 B), count=20 (+20), average=208 B
epoch: 74 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.7 KiB (+23.7 KiB), count=412 (+412), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=17.3 KiB (+17.3 KiB), count=301 (+301), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=13.6 KiB (+13.6 KiB), count=248 (+248), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.3 KiB (+10.3 KiB), count=146 (+146), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=7008 B (+7008 B), count=146 (+146), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 75 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.7 KiB (+23.7 KiB), count=411 (+411), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.7 KiB (+16.7 KiB), count=290 (+290), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=13.2 KiB (+13.2 KiB), count=241 (+241), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10080 B (+10080 B), count=140 (+140), average=72 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=8400 B (+8400 B), count=210 (+210), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6720 B (+6720 B), count=140 (+140), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/functional.py:77: size=2770 B (+2770 B), count=47 (+47), average=59 B
epoch: 76 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=410 (+410), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.6 KiB (+16.6 KiB), count=288 (+288), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=13.6 KiB (+13.6 KiB), count=248 (+248), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.5 KiB (+10.5 KiB), count=150 (+150), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=7200 B (+7200 B), count=150 (+150), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=3422 B (+3422 B), count=58 (+58), average=59 B
epoch: 77 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=410 (+410), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.2 KiB (+16.2 KiB), count=281 (+281), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=13.8 KiB (+13.8 KiB), count=253 (+253), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.5 KiB (+10.5 KiB), count=149 (+149), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=7152 B (+7152 B), count=149 (+149), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=3363 B (+3363 B), count=57 (+57), average=59 B
epoch: 78 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=410 (+410), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.7 KiB (+16.7 KiB), count=289 (+289), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=13.7 KiB (+13.7 KiB), count=251 (+251), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=135 (+135), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.2 KiB (+10.2 KiB), count=145 (+145), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6960 B (+6960 B), count=145 (+145), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=3363 B (+3363 B), count=57 (+57), average=59 B
epoch: 79 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=410 (+410), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.2 KiB (+16.2 KiB), count=281 (+281), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=13.6 KiB (+13.6 KiB), count=249 (+249), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.7 KiB (+10.7 KiB), count=137 (+137), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.1 KiB (+10.1 KiB), count=143 (+143), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6864 B (+6864 B), count=143 (+143), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 80 training seq 6254 of 6255Epoch 80 (time: 97.95434403419495) patience 79/1000 -  loss: 0.33598813379150283 vs best loss: 0.12451329277828335
Stats:  {'mae': 0.0709846243262291, 'r2': 0.01333242654800415, 'pearson_corr': 0.11711318790912628, 'sign_accuracy': 0.8397775701588285}
last learning rate: [0.00023730468750000005]
Predictions: [ 0.12968606 -0.01051681 -0.06715025  0.01457062 -0.45516226 -0.12908334
  0.00109249  0.00378547  0.07527726  0.04273729 -0.03476579 -0.01099522
 -0.17965943 -0.00870438  0.25686136  0.09497148 -0.09490802 -0.1479748
  0.1277011   0.02945194  0.03013217  0.01121642 -0.02387449 -0.24384636
 -0.21061482]
Labels:      [ 0.03172797 -0.01552655 -0.04976128  0.09920722 -0.26288357 -0.12191492
 -0.05245331 -0.07544105  0.03156399  0.03528959  0.00977123 -0.00900743
 -0.27262178  0.03624757  0.10385292  0.10581546 -0.03641085 -0.07890114
 -0.09638985  0.1042879   0.03066839 -0.03789082 -0.06467972 -0.12719823
 -0.16772854]
<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.0 KiB (+23.0 KiB), count=399 (+399), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=18.1 KiB (+18.1 KiB), count=314 (+314), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=13.4 KiB (+13.4 KiB), count=245 (+245), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=8400 B (+8400 B), count=210 (+210), average=40 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=2592 B (+2592 B), count=12 (+12), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=2496 B (+2496 B), count=12 (+12), average=208 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:481: size=2448 B (+2448 B), count=12 (+12), average=204 B
epoch: 81 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=409 (+409), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=18.0 KiB (+18.0 KiB), count=313 (+313), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=13.7 KiB (+13.7 KiB), count=251 (+251), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.7 KiB (+10.7 KiB), count=137 (+137), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10152 B (+10152 B), count=141 (+141), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6768 B (+6768 B), count=141 (+141), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=2800 B (+2800 B), count=70 (+70), average=40 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=2360 B (+2360 B), count=40 (+40), average=59 B
epoch: 82 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.7 KiB (+23.7 KiB), count=411 (+411), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.9 KiB (+16.9 KiB), count=293 (+293), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=13.8 KiB (+13.8 KiB), count=253 (+253), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.3 KiB (+10.3 KiB), count=146 (+146), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=7008 B (+7008 B), count=146 (+146), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 83 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=410 (+410), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.4 KiB (+16.4 KiB), count=285 (+285), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=13.7 KiB (+13.7 KiB), count=250 (+250), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10152 B (+10152 B), count=141 (+141), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6768 B (+6768 B), count=141 (+141), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 84 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=409 (+409), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.6 KiB (+16.6 KiB), count=288 (+288), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=13.9 KiB (+13.9 KiB), count=255 (+255), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.2 KiB (+10.2 KiB), count=145 (+145), average=72 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=8400 B (+8400 B), count=210 (+210), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6960 B (+6960 B), count=145 (+145), average=48 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=3068 B (+3068 B), count=52 (+52), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
epoch: 85 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.5 KiB (+23.5 KiB), count=408 (+408), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.4 KiB (+16.4 KiB), count=284 (+284), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=13.9 KiB (+13.9 KiB), count=255 (+255), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.1 KiB (+10.1 KiB), count=144 (+144), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6912 B (+6912 B), count=144 (+144), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 86 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=409 (+409), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.9 KiB (+16.9 KiB), count=293 (+293), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=13.9 KiB (+13.9 KiB), count=254 (+254), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10224 B (+10224 B), count=142 (+142), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6816 B (+6816 B), count=142 (+142), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 87 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=410 (+410), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=17.3 KiB (+17.3 KiB), count=300 (+300), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=13.8 KiB (+13.8 KiB), count=253 (+253), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.1 KiB (+10.1 KiB), count=143 (+143), average=72 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=8400 B (+8400 B), count=210 (+210), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6864 B (+6864 B), count=143 (+143), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=2592 B (+2592 B), count=12 (+12), average=216 B
epoch: 88 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=409 (+409), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=17.3 KiB (+17.3 KiB), count=300 (+300), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=13.9 KiB (+13.9 KiB), count=254 (+254), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.1 KiB (+10.1 KiB), count=144 (+144), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6912 B (+6912 B), count=144 (+144), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 89 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=409 (+409), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=17.7 KiB (+17.7 KiB), count=307 (+307), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=13.8 KiB (+13.8 KiB), count=253 (+253), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.1 KiB (+10.1 KiB), count=144 (+144), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6912 B (+6912 B), count=144 (+144), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 90 training seq 6254 of 6255Epoch 90 (time: 97.42098689079285) patience 89/1000 -  loss: 0.3271209524519616 vs best loss: 0.12451329277828335
Stats:  {'mae': 0.06991694122552872, 'r2': 0.013990819454193115, 'pearson_corr': 0.12032696604728699, 'sign_accuracy': 0.8441492258422307}
last learning rate: [0.00023730468750000005]
Predictions: [-0.07204778 -0.2505521   0.00172472 -0.06769461 -0.05403395  0.09374138
  0.03981788  0.08575121  0.02486739 -0.15347813 -0.08274583 -0.09796634
 -0.06180245 -0.19131356  0.5067042  -0.07106645 -0.03374289 -0.02226081
  0.0038658   0.073858   -0.14400995  0.00290969 -0.0438349  -0.05940614
 -0.18968987]
Labels:      [-4.1196585e-02 -1.6028115e-01  1.5738958e-03 -6.4416386e-02
 -4.8201852e-02  9.1764763e-02  8.1540689e-02  1.0462852e-01
 -4.2212911e-02 -7.5676739e-02 -7.5776868e-02  3.6256909e+00
 -3.4096796e-02 -1.6104870e-01  5.2179444e-01 -4.8574604e-02
 -9.5550284e-02 -4.4306539e-02  7.8710921e-02  2.7935523e-01
 -1.3623258e-01 -4.9873941e-02  4.5958781e-04 -2.1376796e-03
 -1.1314224e-01]
<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.0 KiB (+23.0 KiB), count=399 (+399), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=17.2 KiB (+17.2 KiB), count=298 (+298), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=13.9 KiB (+13.9 KiB), count=255 (+255), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=135 (+135), average=80 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:481: size=3264 B (+3264 B), count=16 (+16), average=204 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
epoch: 91 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.4 KiB (+23.4 KiB), count=406 (+406), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.4 KiB (+16.4 KiB), count=284 (+284), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=13.9 KiB (+13.9 KiB), count=255 (+255), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=134 (+134), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10224 B (+10224 B), count=142 (+142), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6816 B (+6816 B), count=142 (+142), average=48 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=3068 B (+3068 B), count=52 (+52), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/functional.py:77: size=2593 B (+2593 B), count=44 (+44), average=59 B
epoch: 92 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=410 (+410), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.9 KiB (+16.9 KiB), count=294 (+294), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=13.8 KiB (+13.8 KiB), count=253 (+253), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10080 B (+10080 B), count=140 (+140), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6720 B (+6720 B), count=140 (+140), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 93 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=409 (+409), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.1 KiB (+16.1 KiB), count=279 (+279), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=14.4 KiB (+14.4 KiB), count=264 (+264), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=13.7 KiB (+13.7 KiB), count=350 (+350), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.7 KiB (+10.7 KiB), count=137 (+137), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.5 KiB (+10.5 KiB), count=149 (+149), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=7152 B (+7152 B), count=149 (+149), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=4320 B (+4320 B), count=20 (+20), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=4160 B (+4160 B), count=20 (+20), average=208 B
epoch: 94 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=410 (+410), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.6 KiB (+16.6 KiB), count=288 (+288), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=14.5 KiB (+14.5 KiB), count=266 (+266), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.8 KiB (+10.8 KiB), count=138 (+138), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.6 KiB (+10.6 KiB), count=151 (+151), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=7248 B (+7248 B), count=151 (+151), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 95 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=409 (+409), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=17.1 KiB (+17.1 KiB), count=296 (+296), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=14.7 KiB (+14.7 KiB), count=268 (+268), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.4 KiB (+10.4 KiB), count=148 (+148), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=7104 B (+7104 B), count=148 (+148), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 96 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.5 KiB (+23.5 KiB), count=408 (+408), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=17.0 KiB (+17.0 KiB), count=295 (+295), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=14.9 KiB (+14.9 KiB), count=273 (+273), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.5 KiB (+10.5 KiB), count=150 (+150), average=72 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=8400 B (+8400 B), count=210 (+210), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=7200 B (+7200 B), count=150 (+150), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=2596 B (+2596 B), count=44 (+44), average=59 B
epoch: 97 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=410 (+410), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=17.2 KiB (+17.2 KiB), count=299 (+299), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=14.8 KiB (+14.8 KiB), count=271 (+271), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.7 KiB (+10.7 KiB), count=137 (+137), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.5 KiB (+10.5 KiB), count=149 (+149), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=7152 B (+7152 B), count=149 (+149), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 98 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=409 (+409), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=17.5 KiB (+17.5 KiB), count=303 (+303), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=14.6 KiB (+14.6 KiB), count=267 (+267), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.1 KiB (+10.1 KiB), count=144 (+144), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6912 B (+6912 B), count=144 (+144), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 99 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=410 (+410), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=17.6 KiB (+17.6 KiB), count=306 (+306), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=14.9 KiB (+14.9 KiB), count=273 (+273), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.7 KiB (+10.7 KiB), count=137 (+137), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.3 KiB (+10.3 KiB), count=146 (+146), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=7008 B (+7008 B), count=146 (+146), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 100 training seq 6254 of 6255Epoch 100 (time: 98.70600914955139) patience 99/1000 -  loss: 0.3245558543180262 vs best loss: 0.12451329277828335
Stats:  {'mae': 0.06953481584787369, 'r2': 0.013579368591308594, 'pearson_corr': 0.11819667369127274, 'sign_accuracy': 0.8454032665011266}
last learning rate: [0.00023730468750000005]
Predictions: [ 0.38805836 -0.04935505 -0.04799603 -0.2722445  -0.03740686 -0.0499384
  0.02629294  0.19988488 -0.10919122 -0.18046279 -0.25969124 -0.07457425
  0.02073989 -0.09712728 -0.03904343  0.03728103 -0.06112355 -0.01185639
 -0.21949466 -0.04276036  0.11021522 -0.01612337  0.14810187 -0.12926072
 -0.1550488 ]
Labels:      [ 0.35415578 -0.12384514 -0.05495289 -0.17899235 -0.05245331 -0.10298476
  0.0300488   0.12993096 -0.10969715 -0.14199352 -0.24529397 -0.02994701
  0.03263168 -0.1084962   0.00577329 -0.02409296 -0.0637326   0.03812724
 -0.20053047 -0.08787274  0.01641682 -0.02684973  0.06816779 -0.10120898
 -0.25284913]
<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=22.9 KiB (+22.9 KiB), count=397 (+397), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=17.1 KiB (+17.1 KiB), count=297 (+297), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=14.7 KiB (+14.7 KiB), count=268 (+268), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=135 (+135), average=80 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:481: size=3264 B (+3264 B), count=16 (+16), average=204 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
epoch: 101 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.5 KiB (+23.5 KiB), count=407 (+407), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.8 KiB (+16.8 KiB), count=291 (+291), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=14.5 KiB (+14.5 KiB), count=266 (+266), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=134 (+134), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=9936 B (+9936 B), count=138 (+138), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6624 B (+6624 B), count=138 (+138), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=2800 B (+2800 B), count=70 (+70), average=40 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=2773 B (+2773 B), count=47 (+47), average=59 B
epoch: 102 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.5 KiB (+23.5 KiB), count=408 (+408), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.6 KiB (+16.6 KiB), count=288 (+288), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=14.9 KiB (+14.9 KiB), count=273 (+273), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=13.7 KiB (+13.7 KiB), count=350 (+350), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.7 KiB (+10.7 KiB), count=137 (+137), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.4 KiB (+10.4 KiB), count=148 (+148), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=7104 B (+7104 B), count=148 (+148), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=4320 B (+4320 B), count=20 (+20), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=4160 B (+4160 B), count=20 (+20), average=208 B
epoch: 103 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.5 KiB (+23.5 KiB), count=408 (+408), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=17.2 KiB (+17.2 KiB), count=299 (+299), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=15.0 KiB (+15.0 KiB), count=274 (+274), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.3 KiB (+10.3 KiB), count=147 (+147), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=7056 B (+7056 B), count=147 (+147), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=2800 B (+2800 B), count=70 (+70), average=40 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=2596 B (+2596 B), count=44 (+44), average=59 B
epoch: 104 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=409 (+409), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.6 KiB (+16.6 KiB), count=288 (+288), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=15.0 KiB (+15.0 KiB), count=275 (+275), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.4 KiB (+10.4 KiB), count=133 (+133), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.2 KiB (+10.2 KiB), count=145 (+145), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6960 B (+6960 B), count=145 (+145), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 105 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=409 (+409), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.3 KiB (+16.3 KiB), count=283 (+283), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=15.1 KiB (+15.1 KiB), count=277 (+277), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=135 (+135), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.5 KiB (+10.5 KiB), count=149 (+149), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=7152 B (+7152 B), count=149 (+149), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 106 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=410 (+410), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.5 KiB (+16.5 KiB), count=287 (+287), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=15.1 KiB (+15.1 KiB), count=276 (+276), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.5 KiB (+10.5 KiB), count=149 (+149), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=7152 B (+7152 B), count=149 (+149), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 107 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=410 (+410), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.5 KiB (+16.5 KiB), count=287 (+287), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=15.1 KiB (+15.1 KiB), count=276 (+276), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.5 KiB (+10.5 KiB), count=135 (+135), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.2 KiB (+10.2 KiB), count=145 (+145), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6960 B (+6960 B), count=145 (+145), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 108 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=409 (+409), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.5 KiB (+16.5 KiB), count=287 (+287), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=15.0 KiB (+15.0 KiB), count=274 (+274), average=56 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10.3 KiB (+10.3 KiB), count=147 (+147), average=72 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=7056 B (+7056 B), count=147 (+147), average=48 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
epoch: 109 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.6 KiB (+23.6 KiB), count=409 (+409), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.5 KiB (+16.5 KiB), count=287 (+287), average=59 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:558: size=14.8 KiB (+14.8 KiB), count=271 (+271), average=56 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:115: size=10.6 KiB (+10.6 KiB), count=136 (+136), average=80 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:129: size=10224 B (+10224 B), count=142 (+142), average=72 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=8400 B (+8400 B), count=210 (+210), average=40 B
/opt/homebrew/Cellar/python@3.11/3.11.11/Frameworks/Python.framework/Versions/3.11/lib/python3.11/tracemalloc.py:498: size=6816 B (+6816 B), count=142 (+142), average=48 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=2832 B (+2832 B), count=48 (+48), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
epoch: 110 training seq 6254 of 6255Epoch 110 (time: 98.60715699195862) patience 109/1000 -  loss: 0.3205780958741636 vs best loss: 0.12451329277828335
Stats:  {'mae': 0.06911717355251312, 'r2': 0.013584434986114502, 'pearson_corr': 0.11820068210363388, 'sign_accuracy': 0.8473817529589863}
last learning rate: [0.00023730468750000005]
Predictions: [-0.0718566  -0.0522601   0.01304352 -0.00376742 -0.16190076 -0.04689923
  0.00720658  0.01337168 -0.12877391 -0.0818632  -0.16291045 -0.05690128
  0.0694928  -0.07753675  0.04415974 -0.07757392 -0.15031752  0.08861699
 -0.05392973  0.49074337 -0.35645068 -0.07563639 -0.03341439 -0.02854282
 -0.02227083]
Labels:      [-0.10636798  0.00919388 -0.02192481 -0.04614433 -0.08995303 -0.04704569
  0.0779764   0.10558456 -0.11106306 -0.04937114 -0.25284913 -0.06598494
  0.0972035  -0.09868871 -0.0194538  -0.05156754 -0.18722996  0.13532256
 -0.01260925  0.5003969  -0.37726107  0.09284312  0.05095536 -0.01461127
  0.08312615]
<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=22.9 KiB (+22.9 KiB), count=397 (+397), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=15.6 KiB (+15.6 KiB), count=270 (+270), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=10.9 KiB (+10.9 KiB), count=280 (+280), average=40 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:584: size=3456 B (+3456 B), count=16 (+16), average=216 B
/Users/jmordetsky/alfred/src/alfred/models/vanilla.py:102: size=3363 B (+3363 B), count=57 (+57), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:677: size=3328 B (+3328 B), count=16 (+16), average=208 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:481: size=3264 B (+3264 B), count=16 (+16), average=204 B
epoch: 111 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.7 KiB (+23.7 KiB), count=412 (+412), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.8 KiB (+16.8 KiB), count=291 (+291), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/simplejson/encoder.py:370: size=2800 B (+2800 B), count=70 (+70), average=40 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/functional.py:77: size=2655 B (+2655 B), count=45 (+45), average=59 B
epoch: 112 training seq 6254 of 6255<frozen abc>:123: size=38.6 KiB (+38.6 KiB), count=219 (+219), average=180 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/_ops.py:1061: size=23.7 KiB (+23.7 KiB), count=412 (+412), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py:214: size=16.4 KiB (+16.4 KiB), count=284 (+284), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/optim/adam.py:131: size=2816 B (+2816 B), count=32 (+32), average=88 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/functional.py:77: size=2711 B (+2711 B), count=46 (+46), average=59 B
/Users/jmordetsky/alfred/venv/lib/python3.11/site-packages/torch/nn/modules/rnn.py:917: size=2655 B (+2655 B), count=45 (+45), average=59 B
epoch: 113 training seq 6254 of 6255