import pandas as pd
import sys
import os

# Add GlossBERT folder to path
sys.path.append("GlossBERT")

from run_infer_demo_sent_cls_ws import infer
import argparse

# Arguments exactly like demo script
args = argparse.Namespace(
    bert_model="./GlossBERT/Sent_CLS_WS",
    no_cuda=False
)

df = pd.read_csv("semcor_glossbert.csv")
results = []

import time

start = time.time()

# Start with first 100 rows only
for i, row in df.head(100).iterrows():

    sentence = row["sentence"]
    lemma = row["lemma"]
    gold_sense = row["gold_sense"]

    start_id = int(row["target_start_id"])
    end_id = int(row["target_end_id"])

    try:

        prediction = infer(
            sentence,
            start_id,
            end_id,
            lemma,
            args
        )
        print("Prediction returned:", prediction)
        results.append([
            sentence,
            row["target_word"],
            gold_sense,
            prediction
        ])

    except Exception as e:

        print(f"Error on {lemma}: {e}")

results_df = pd.DataFrame(
    results,
    columns=[
        "sentence",
        "target_word",
        "gold_sense",
        "glossbert_prediction"
    ]
)

results_df.to_csv(
    "glossbert_predictions.csv",
    index=False
)
end = time.time()

print("Time:", end - start)
print("\nSaved glossbert_predictions.csv")
print(results_df.head())