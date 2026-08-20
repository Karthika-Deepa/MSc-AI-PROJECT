from transformers import pipeline

# Load pretrained BERT masked language model
classifier = pipeline(
    "fill-mask",
    model="bert-base-uncased"
)

# Test sentence
sentence = "The fisherman sat on the river [MASK]."

# Predict contextual word
result = classifier(sentence)

print("\nEWISER-style Transformer WSD Output:\n")

for r in result[:5]:
    print(f"Prediction : {r['token_str']}")
    print(f"Score      : {r['score']}")
    print(f"Sentence   : {r['sequence']}")
    print("-" * 40)
    