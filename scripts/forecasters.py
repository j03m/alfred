from next_gen import train_tr_forecaster

train_tr_forecaster(model_path="./models",
                 model_prefix="tranformer_forecaster_",
                 training_data_path="./data/AAPL_diffs.csv")