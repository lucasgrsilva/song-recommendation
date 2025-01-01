import pandas as pd
import pickle
from fpgrowth_py import fpgrowth
from datetime import datetime

# Load datasets
spotfi_1 = pd.read_csv('2023_spotify_ds1.csv')
spotfi_2 = pd.read_csv('2023_spotify_ds2.csv')

data = pd.concat([spotfi_1, spotfi_2], ignore_index=True)

track_transactions = data.groupby('pid')['track_name'].apply(list).tolist()

# Generate frequent itemsets and association rules using fpgrowth_py
min_support = 0.05
min_confidence = 0.2

track_freq_itemsets, track_rules = fpgrowth(track_transactions, minSupRatio=min_support, minConf=min_confidence)

track_popularity = data['track_name'].value_counts()

model_metadata = {
    "version": "1.0.1",
    "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

model_data = {
    "metadata": model_metadata,
    "track_frequent_itemsets": track_freq_itemsets,
    "track_rules": track_rules,
    "track_popularity": track_popularity
}

with open('recommendation_model.pickle', 'wb') as model_file:
    pickle.dump(model_data, model_file)

distinct_antecedents = set()
for rule in track_rules:
    distinct_antecedents.update(rule[0])

print(f"Model saved successfully with rules for {len(distinct_antecedents)} distinct songs")
