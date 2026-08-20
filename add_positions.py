import pandas as pd

df = pd.read_csv("semcor_fulldataset.csv")

start_ids = []
end_ids = []

for i, row in df.iterrows():

    sentence = row["sentence"]
    target = row["target_word"]

    words = sentence.split()

    try:
        pos = words.index(target)

        start_ids.append(pos)
        end_ids.append(pos+1)

    except:
        start_ids.append(-1)
        end_ids.append(-1)

df["target_start_id"] = start_ids
df["target_end_id"] = end_ids

df.to_csv("semcor_glossbert.csv", index=False)

print(df.head())
print("\nSaved as semcor_glossbert.csv")