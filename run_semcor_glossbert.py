import pandas as pd
import sys
import os
import time

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
print("Total rows:", len(df))

# CHANGE THESE FOR EACH CHUNK
START = 200000
END = 229533

chunk = df.iloc[START:END]
results = []
start = time.time()

for i, row in chunk.iterrows():
     # Progress every 100 rows
    if (i - START) % 100 == 0:
        print(f"Processed {i-START}/{END-START}")

    sentence = row["sentence"]
    lemma = row["lemma"]
    gold_sense = row["gold_sense"]

    start_id = int(row["target_start_id"])
    end_id = int(row["target_end_id"])
   
   # Skip failed alignments
    if start_id == -1:
        continue

    try:
        # if i % 100 == 0:
        #  print(f"Processed {i} rows...")
        prediction = infer(
            sentence,
            start_id,
            end_id,
            lemma,
            args
        )
        # print("Prediction returned:", prediction)
        results.append([
            sentence,
            row["target_word"],
            gold_sense,
            prediction
        ])
                # Save checkpoint every 100 rows
        if (i - START) % 100 == 0 and (i - START) > 0:

            checkpoint_file = f"glossbert_checkpoint_{START}_{END}.csv"

            pd.DataFrame(
                results,
                columns=[
                    "sentence",
                    "target_word",
                    "gold_sense",
                    "glossbert_prediction"
                ]
            ).to_csv(
                checkpoint_file,
                index=False
            )

            print(f"Checkpoint saved at row {i}")

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

filename = f"glossbert_{START}_{END}.csv"

results_df.to_csv(
    filename,
    index=False
)
end = time.time()

print("Completed Successfully")
print("Rows Processed:", len(results_df))
print("Output File:", filename)
print("Time Taken (seconds):", end- start)