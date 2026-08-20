import pandas as pd

# Load GlossBERT results  
df_gb = pd.read_csv("glossbert_full.csv")
df_gb['correct'] = df_gb['glossbert_prediction'] == df_gb['gold_sense']

# GlossBERT: top 20 words with most errors
gb_errors = df_gb[~df_gb['correct']]
gb_top20 = gb_errors.groupby('target_word').size().sort_values(ascending=False).head(20)
print("\nGlossBERT Top 20 Most Misclassified Words")
for word, count in gb_top20.items():
    total = (df_gb['target_word'] == word).sum()
    err_rate = count / total * 100
    print(f"  {word:<15} errors: {count:>5}  /  total: {total:>6}  ({err_rate:.1f}% error rate)")