from sacred import Experiment
from sacred.observers import MongoObserver
from alfred.utils import trim_timerange, set_deterministic
from alfred.easy import trainer, evaler
from alfred.metadata import ExperimentSelector, TickerCategories
from alfred.utils import MongoConnectionStrings
from alfred.model_persistence import crc32_columns
from alfred.model_metrics import BCEAccumulator, RegressionAccumulator, HuberWithSignPenalty, MSEWithSignPenalty, \
    SignErrorRatioLoss

import torch.nn as nn

import argparse

gbl_args = None

set_deterministic(0)

connection_data = MongoConnectionStrings()
connection_data.get_mongo_client()  # don't use this but it will force the right connection strings

DB = 'sacred_db'
MONGO = connection_data.connection_string()

# I hate that this is global
experiment_namespace = "alfred-experiments-2"
ex = Experiment(experiment_namespace)
ex.add_config({'token': experiment_namespace})
ex.observers.append(MongoObserver(
    url=MONGO,
    db_name=DB
))


@ex.config
def config():
    model_name = None,
    size = None,
    sequence_length = None



def build_experiment_descriptor_key(input_config):
    model_token = f"{gbl_args.category}:{input_config['model_name']}"
    size = input_config["size"]
    sequence_length = input_config["sequence_length"]
    return f"{model_token}:{size}:{sequence_length}"


@ex.main
def run_experiment(model_name, size, sequence_length):
    ticker_metadata = TickerCategories(gbl_args.ticker_categories_file)
    training_tickers = ticker_metadata.get(["training"])
    eval_tickers = ticker_metadata.get(["training"])
    training_files = []
    eval_files = []
    for ticker in training_tickers:
        training_files.append(f"data/{ticker}{gbl_args.file_post_fix}.csv")

    for ticker in eval_tickers:
        eval_files.append(f"data/{ticker}{gbl_args.file_post_fix}.csv")

    loss_function = None
    loss = gbl_args.loss
    if loss == "bce":
        loss_function = nn.BCELoss()
    elif loss == "mse":
        loss_function = nn.MSELoss()
    elif loss == "huber-sign":
        loss_function = HuberWithSignPenalty()
    elif loss == "mse-sign":
        loss_function = MSEWithSignPenalty()
    elif loss == "sign-ratio":
        loss_function = SignErrorRatioLoss()
    return_data = {}
    if not gbl_args.eval_only:
        print("Starting easy trainer")
        training_loss, training_stats, time_per_epoch = trainer(
            category=gbl_args.category,
            augment_func=lambda df: trim_timerange(df, min_date=gbl_args.min_date, max_date=gbl_args.max_date),
            files=training_files,
            patience=gbl_args.patience,
            verbose=True,
            model_size=size,
            model_name=model_name,
            labels=[gbl_args.label],
            epochs=gbl_args.epochs,
            loss_function=loss_function,
            seq_len=sequence_length,
            stat_accumulator=BCEAccumulator() if loss == "bce" else RegressionAccumulator())

        ex.log_scalar('training_loss', training_loss)
        return_data['training_loss'] = training_loss

        ex.log_scalar('time_per_epoch', time_per_epoch)
        return_data['time_per_epoch'] = time_per_epoch

        for key, value in training_stats.items():
            metric_name = f'training_{key}'
            ex.log_scalar(metric_name, value)
            return_data[metric_name] = value
    else:
        print("skip training, eval only...")

    eval_loss, eval_stats = evaler(category=gbl_args.category,
                    augment_func=lambda df: trim_timerange(df, min_date=gbl_args.min_date, max_date=gbl_args.max_date),
                    model_size=size,
                    model_name=model_name,
                    seq_len=sequence_length,
                    files=eval_files,
                    labels=[gbl_args.label],
                    loss_function=loss_function,
                    stat_accumulator=BCEAccumulator() if gbl_args.loss == "bce" else RegressionAccumulator())


    ex.log_scalar('loss_type', gbl_args.loss)
    return_data['loss_type'] = gbl_args.loss

    ex.log_scalar('eval_loss', eval_loss)
    return_data['eval_loss'] = eval_loss

    for key, value in eval_stats.items():
        metric_name = f'eval_{key}'  # Corrected from 'training_{key}'
        ex.log_scalar(metric_name, value)
        return_data[metric_name] = value

    return return_data

def main(args):
    selector = ExperimentSelector(index_file=args.index_file, mongo=MONGO, db=DB)
    experiments = selector.get(include_ranges=args.include, exclude_ranges=args.exclude)

    # get a list of past or in flight experiments
    past_experiments = selector.get_current_state(experiment_namespace, build_experiment_descriptor_key)

    # Run the experiments using Sacred
    for experiment in experiments:
        if experiment:
            key = build_experiment_descriptor_key(experiment)
            if key not in past_experiments:
                ex.run(config_updates=experiment)
                # update the list in case another machine is running (poor man's update)
                past_experiments = selector.get_current_state("analysts", build_experiment_descriptor_key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run selected experiments using Sacred.")
    parser.add_argument("--index-file", type=str, default="./metadata/experiment-index.json",
                        help="Path to the JSON file containing indexed experiments")
    parser.add_argument("--test-symbol", type=str,
                        help="If supplied, will circumvent the metadate file and only test this symbol")
    parser.add_argument("--column-file", type=str, default="./metadata/column-descriptors.json",
                        help="Path to the JSON file containing indexed experiments")
    parser.add_argument("--ticker-categories-file", type=str, default="./metadata/basic-tickers.json",
                        help="Path to the JSON file containing tickers for the experiments")
    parser.add_argument("--include", type=str, default="",
                        help="Ranges of experiments to include (e.g., 1-5,10-15)")
    parser.add_argument("--exclude", type=str, default="",
                        help="Ranges of experiments to exclude (e.g., 4-5,8)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed is combined with a ticker to produce a consistent random training and eval period")
    parser.add_argument('--min_date', type=str, default="2004-03-31",
                        help='Minimum date for timerange trimming (YYYY-MM-DD)')
    parser.add_argument('--max_date', type=str, default=None, help='Maximum date for timerange trimming (YYYY-MM-DD)')

    parser.add_argument("--epochs", type=int, default=5000,
                        help="number of epochs to train")
    parser.add_argument("--patience", type=int, default=1000,
                        help="when to stop training after patience epochs of no improvements")
    parser.add_argument("--model-path", type=str, default='./models', help="where to store models and best loss data")
    parser.add_argument("--metadata-path", type=str, default='./metadata', help="experiment descriptors live here")
    parser.add_argument("--category", type=str, default='easy_experiments', help="Use this to augment groups of experiments")
    parser.add_argument('--file-post-fix', type=str, default="_quarterly_magnitude",
                        help='assumes data/[ticker][args.file_post_fix].csv as data to use')
    parser.add_argument('--loss', choices=["bce", "mse", "huber-sign", "mse-sign", "sign-ratio"], default="huber-sign",
                        help='loss function')
    parser.add_argument('--label', type=str, default="PM",
                        help='label column')
    parser.add_argument('--eval-only',
                        action='store_true',
                        default=False,
                        help='do not train')

    gbl_args = parser.parse_args()
    main(gbl_args)
