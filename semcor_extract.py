# This code will:
# read semcor.xml
# reconstruct sentences
# extract annotated words
# save CSV

import xml.etree.ElementTree as ET
import csv

# Path to SemCor XML
xml_path = r"C:\Users\KarthikaDeepa\WSD_SYSTEMS\UFSAC\java\data\corpora\semcor.xml"

# Parse XML
tree = ET.parse(xml_path)
root = tree.getroot()

# Output CSV
output_file = "semcor_fulldataset.csv"

rows = []

# Iterate through sentences
for sentence in root.iter("sentence"):

    words = []
    annotated_words = []

    # Build sentence
    for word in sentence.findall("word"):

        surface = word.attrib.get("surface_form", "")
        words.append(surface)

    full_sentence = " ".join(words)

    # Extract annotated target words
    for word in sentence.findall("word"):

        if "wn30_key" in word.attrib:

            target_word = word.attrib.get("surface_form", "")
            lemma = word.attrib.get("lemma", "")
            pos = word.attrib.get("pos", "")
            gold_sense = word.attrib.get("wn30_key", "")

            rows.append([
                full_sentence,
                target_word,
                lemma,
                pos,
                gold_sense
            ])

# Write CSV
with open(output_file, mode="w", newline="", encoding="utf-8") as file:

    writer = csv.writer(file)

    writer.writerow([
        "sentence",
        "target_word",
        "lemma",
        "pos",
        "gold_sense"
    ])

    writer.writerows(rows)

print(f"Saved {len(rows)} rows to {output_file}")