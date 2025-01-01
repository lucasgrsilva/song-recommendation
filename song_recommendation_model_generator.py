import pandas as pd
import pickle
from fpgrowth_py import fpgrowth

DATASET_1 = "2023_spotify_ds1.csv"
DATASET_2 = "2023_spotify_ds2.csv"
MODEL_FILE = "recommendation_model.pickle"

def load_data(file_path):
    data = pd.read_csv(file_path)
    transactions = data.groupby('pid')['track_name'].apply(list).tolist()
    return transactions

def generate_rules(transactions, min_sup=0.1, min_conf=0.3):
    freq_item_set, rules = fpgrowth(transactions, minSupRatio=min_sup, minConf=min_conf)
    return freq_item_set, rules

print(f"Loading data from {DATASET_1}...")
transactions_ds1 = load_data(DATASET_1)

print("Training model with dataset 1...")
freq_item_set_ds1, rules_ds1 = generate_rules(transactions_ds1)

model = {
    "freq_item_set": freq_item_set_ds1,
    "rules": rules_ds1,
    "track_popularity": {}, 
}
print(f"Initial model trained with {len(rules_ds1)} rules.")

if DATASET_2:
    print(f"Loading data from {DATASET_2}...")
    transactions_ds2 = load_data(DATASET_2)
    
    print("Updating model with dataset 2...")
    freq_item_set_ds2, rules_ds2 = generate_rules(transactions_ds2)
    
    model["rules"].extend(rules_ds2)
    model["freq_item_set"].extend(freq_item_set_ds2)
    print(f"Updated model with {len(rules_ds2)} additional rules.")

print(f"Saving model to {MODEL_FILE}...")
with open(MODEL_FILE, "wb") as model_file:
    pickle.dump(model, model_file)

print(f"Model saved! Total rules: {len(model['rules'])}")