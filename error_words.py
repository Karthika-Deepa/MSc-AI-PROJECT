import pandas as pd

df = pd.read_csv("glossbert_full.csv")
df['correct'] = df['glossbert_prediction'] == df['gold_sense']

TARGET = 'have'   # change each time

errors = df[(df['target_word'] == TARGET) & (~df['correct'])].copy()

print(f"'{TARGET}' — {len(errors)} GlossBERT errors")
print(f"\nGold senses (most common):")
print(errors['gold_sense'].value_counts().head(8))
print(f"\nPredicted senses (most common):")
print(errors['glossbert_prediction'].value_counts().head(8))
print(f"\n=== Sample sentences ===")
for _, row in errors.head(12).iterrows():
    print(f"\nSentence: {row['sentence'][:120]}")
    print(f"  Gold: {row['gold_sense']}")
    print(f"  Pred: {row['glossbert_prediction']}")