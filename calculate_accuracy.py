import pandas as pd

df = pd.read_csv("glossbert_full.csv")

correct = (df["gold_sense"] == df["glossbert_prediction"]).sum()
total = len(df)


accuracy = correct / total * 100

print("Correct:", correct)
print("Total:", total)
print("Accuracy:", round(accuracy, 2), "%")