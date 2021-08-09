import argparse
import pandas as pd

###########################################
# Command line interface
default_dir = "datasets"

parser = argparse.ArgumentParser(description=r"This script will test a model's performance with ICNN dataset")
parser.add_argument('--folder',
        type = str,
        help = f'Path for desired dataset folder. Default: {default_dir}. '
        'The dataset folder contains csv files for the complete, training and testing dataset files',
        default = default_dir
        )
args = parser.parse_args()
###########################################

def check_models(models):
    for model in models:
        complete_set = f"{args.folder}/{model}/dataframe.csv"
        test_set = f"{args.folder}/{model}/test.csv"
        try:
            test_df = pd.read_csv(test_set)
            complete_df = pd.read_csv(complete_set)
        except:
            print(f"Skipping(No file found): {test_set}")
            continue
        print(f"{complete_df.duplicated(subset='sequence', keep='first').sum()} duplicate sequences on complete dataset: {complete_set}")
        print(f"{test_df.duplicated(subset='sequence', keep='first').sum()} duplicate sequences on test dataset: {test_set}")
        other_models = models
        for other_model in other_models:
            train_set = f"{args.folder}/{other_model}/train.csv"
            try:
                train_df = pd.read_csv(train_set)
            except:
                print(f"Skipping(No file found): {test_set}")
                continue
            print(f"{test_df.duplicated(subset='sequence', keep='first').sum()} duplicate sequences on train dataset: {train_set}")
            df_len = len(train_df)
            cond = train_df['sequence'].isin(test_df['sequence'])
            data_cond = train_df[cond]
            if len(data_cond) > 0:
                print(data_cond['sequence'])
            train_df.drop(data_cond.index, inplace = True)
            print(f"{df_len - len(train_df)} overlapping sequences on (test {model} - train {other_model}) pair")

models = ["dprom_complete", "cnnprom_complete", "icnn_complete"]
check_models(models)
models = ["dprom_TATA", "cnnprom_TATA"]
check_models(models)
models = ["dprom_nonTATA", "cnnprom_nonTATA"]
check_models(models)