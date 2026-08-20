import pandas as pd

TOP_N = 20 

# FILE PATH
gloss_path = r"C:/Users/KarthikaDeepa/WSD_SYSTEMS/TRANSFORMER_WSD/glossbert_full.csv"

# LOAD FILE
df = pd.read_csv(gloss_path)

print("Columns in GlossBERT file:")
print(df.columns.tolist())

# BASIC CLEANUP
# Use target_word as the grouping field because the file
# does not contain a separate lemma column.
df = df[df["target_word"].notna()].copy()
df["target_word"] = df["target_word"].astype(str)

# MARK CORRECT / INCORRECT
# GlossBERT predicts sense keys directly, so compare
# glossbert_prediction with gold_sense.
df["correct"] = df["glossbert_prediction"] == df["gold_sense"]

# Keep only wrong rows
errors = df[~df["correct"]].copy()

# A) TOP-N MOST MISCLASSIFIED TARGET WORDS BY ERROR COUNT
print("\n" + "="*80)
print(f"GlossBERT - Top {TOP_N} Most Misclassified Target Words by ERROR COUNT")
print("="*80)

top_by_count = (
    errors.groupby("target_word")
    .size()
    .sort_values(ascending=False)
    .head(TOP_N)
)

print(f"Number of words shown: {len(top_by_count)}\n")

for word, error_count in top_by_count.items():
    total_count = (df["target_word"] == word).sum()
    error_rate = (error_count / total_count) * 100 if total_count > 0 else 0

    print(
        f"{word:<20} "
        f"errors: {error_count:>5}  /  "
        f"total: {total_count:>6}  "
        f"({error_rate:.1f}% error rate)"
    )

# B) TOP-N MOST MISCLASSIFIED TARGET WORDS BY ERROR RATE
print("\n" + "="*80)
print(f"GlossBERT - Top {TOP_N} Most Misclassified Target Words by ERROR RATE")
print("="*80)

# Total count per target word
total_per_word = df.groupby("target_word").size().rename("total_count")

# Error count per target word
error_per_word = errors.groupby("target_word").size().rename("error_count")

# Merge
summary = pd.concat([total_per_word, error_per_word], axis=1).fillna(0)
summary["error_count"] = summary["error_count"].astype(int)

# Error rate
summary["error_rate"] = summary["error_count"] / summary["total_count"] * 100

# Apply minimum frequency filter so rare words don't dominate
MIN_FREQ = 10
summary = summary[summary["total_count"] >= MIN_FREQ].copy()

top_by_rate = (
    summary.sort_values(
        by=["error_rate", "error_count"],
        ascending=[False, False]
    )
    .head(TOP_N)
)

print(f"Minimum frequency filter used: {MIN_FREQ}")
print(f"Number of words shown: {len(top_by_rate)}\n")

for word, row in top_by_rate.iterrows():
    print(
        f"{word:<20} "
        f"errors: {int(row['error_count']):>5}  /  "
        f"total: {int(row['total_count']):>6}  "
        f"({row['error_rate']:.1f}% error rate)"
    )