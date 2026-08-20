import pandas as pd
from nltk.corpus import wordnet as wn

# Load SemCor CSV
df = pd.read_csv("semcor_sample.csv")

# Take first 5 rows
sample = df.head(5)

for index, row in sample.iterrows():

    word = row['lemma']
    sentence = row['sentence']
    gold_sense = row['gold_sense']

    print("\n" + "="*80)
    print("Sentence:")
    print(sentence)

    print("\nTarget Word:")
    print(word)

    print("\nGold Sense:")
    print(gold_sense)

    print("\nCandidate WordNet Senses:\n")

    synsets = wn.synsets(word)

    for i, syn in enumerate(synsets):

        print(f"Sense {i+1}")
        print("Synset :", syn.name())
        print("Definition :", syn.definition())
        print("-"*50)