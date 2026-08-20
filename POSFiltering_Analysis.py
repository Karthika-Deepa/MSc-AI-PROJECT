import pandas as pd
import re

# Load GlossBERT results
df_gb = pd.read_csv("glossbert_full.csv")
df_gb['correct'] = df_gb['glossbert_prediction'] == df_gb['gold_sense']

def get_wn_pos_from_sk(sense_key):
    """Extract WordNet POS from a sense key (both gold and predicted)"""
    if pd.isna(sense_key) or sense_key == '':
        return None
    m = re.match(r'^[^%]+%(\d):', str(sense_key))
    if m:
        return {'1':'n', '2':'v', '3':'a', '4':'r', '5':'a'}.get(m.group(1))
    return None

# GlossBERT outputs sense keys directly so we extract POS from the sense key itself
df_gb['gold_wn_pos'] = df_gb['gold_sense'].apply(get_wn_pos_from_sk)
df_gb['pred_wn_pos'] = df_gb['glossbert_prediction'].apply(get_wn_pos_from_sk)

# Find cross-POS errors in GlossBERT
gb_cross = df_gb[
    df_gb['glossbert_prediction'].notna() &
    (df_gb['gold_wn_pos'] != df_gb['pred_wn_pos']) &
    df_gb['gold_wn_pos'].notna() &
    df_gb['pred_wn_pos'].notna() &
    ~df_gb['correct']   # only wrong predictions
]

print(f"GlossBERT total errors: {(~df_gb['correct']).sum():,}")
print(f"GlossBERT cross-POS errors: {len(gb_cross):,} ({len(gb_cross)/(~df_gb['correct']).sum()*100:.1f}% of errors)")
print(f"NOTE: CANNOT fix GlossBERT's output - this is analysis only.")
print(f"GlossBERT outputs sense keys directly; there is no post-processing step to intercept.")

print("\nMost common cross-POS patterns:")
print(gb_cross.groupby(['gold_wn_pos','pred_wn_pos']).size()
      .sort_values(ascending=False).head(10))

print("\nSample cross-POS errors in GlossBERT:")
print(gb_cross[['target_word','gold_sense','glossbert_prediction',
                'gold_wn_pos','pred_wn_pos']].head(8).to_string())