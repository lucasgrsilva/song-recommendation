import pandas as pd
import os
import argparse
import pickle
from fpgrowth_py import fpgrowth

MODEL_FILE = "/app/datasets/recommendation_model.pickle"

def load_data(file_path):
    data = pd.read_csv(file_path)
    transactions = data.groupby('pid')['track_name'].apply(list).tolist()
    return transactions

def generate_rules(transactions, min_sup=0.1, min_conf=0.3):
    freq_item_set, rules = fpgrowth(transactions, minSupRatio=min_sup, minConf=min_conf)
    return freq_item_set, rules

def main(dataset_path):
    print(f"Dataset Path: {dataset_path}")

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        return

    print(f"Loading data from {dataset_path}...")
    transactions_ds = load_data(dataset_path)

    print("Training model with dataset...")
    freq_item_set_ds, rules_ds = generate_rules(transactions_ds)

    model = {
        "freq_item_set": freq_item_set_ds,
        "rules": rules_ds,
        "track_popularity": {}, 
    }
    print(f"Model trained with {len(rules_ds)} rules.")

    print(f"Saving model to {MODEL_FILE}...")
    with open(MODEL_FILE, "wb") as model_file:
        pickle.dump(model, model_file)

    print(f"Model saved! Total rules: {len(model['rules'])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, required=True, help="Path to the dataset")
    args = parser.parse_args()
    main(args.dataset_path)