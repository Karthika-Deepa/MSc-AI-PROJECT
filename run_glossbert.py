import pandas as pd
from nltk.corpus import wordnet as wn
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load GlossBERT
tokenizer = BertTokenizer.from_pretrained("GlossBERT")
model = BertModel.from_pretrained("GlossBERT")

# Load SemCor CSV
df = pd.read_csv("semcor_sample.csv")

results = []

# Take first 10 rows initially
sample = df.head(10)

for index, row in sample.iterrows():

    sentence = row['sentence']
    target_word = row['lemma']
    gold_sense = row['gold_sense']

    synsets = wn.synsets(target_word)

    if not synsets:
        continue

    # Sentence embedding
    sent_inputs = tokenizer(sentence, return_tensors="pt", truncation=True)

    with torch.no_grad():
        sent_output = model(**sent_inputs)

    sent_embedding = sent_output.last_hidden_state.mean(dim=1)

    best_score = -1
    best_synset = None

    # Compare with each gloss
    for syn in synsets:

        gloss = syn.definition()

        gloss_inputs = tokenizer(gloss, return_tensors="pt", truncation=True)

        with torch.no_grad():
            gloss_output = model(**gloss_inputs)

        gloss_embedding = gloss_output.last_hidden_state.mean(dim=1)

        score = cosine_similarity(
            sent_embedding.numpy(),
            gloss_embedding.numpy()
        )[0][0]

        if score > best_score:
            best_score = score
            best_synset = syn.name()

    results.append([
        sentence,
        target_word,
        gold_sense,
        best_synset
    ])

# Save results
results_df = pd.DataFrame(results, columns=[
    "sentence",
    "target_word",
    "gold_sense",
    "glossbert_prediction"
])

results_df.to_csv("glossbert_results.csv", index=False)

print("\nGlossBERT predictions saved!")
print(results_df.head())