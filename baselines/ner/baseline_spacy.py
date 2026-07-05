import spacy

# Load the largest Portuguese spaCy model
try:
    nlp = spacy.load("pt_core_news_lg")
except OSError:
    print("Error: Please install the model using 'python -m spacy download pt_core_news_lg'")
    exit()

# Example text
texto_ata = "No dia 09 de Agosto de 2015, a Sra. Ana Laura Torres, residente na Av. Eva Jesus, chegou na sua Mercedes-Benz com a matrícula 76-41-46."


def main():
    print("--- BASELINE: SPACY (pt_core_news_lg) ---")

    # Process the text to extract entities
    doc = nlp(texto_ata)

    print("Detected entities:")
    for ent in doc.ents:
        # spaCy uses labels like PER (Person), LOC (Location), ORG (Organization), MISC (Miscellaneous)
        print(f" - [{ent.label_}]: {ent.text}")

    # Anonymize (Replace text with the exact entity tag)
    texto_anonimizado = texto_ata

    # We iterate in reverse to avoid index shifting when replacing strings of different lengths
    for ent in reversed(doc.ents):
        start = ent.start_char
        end = ent.end_char
        tag = f"<{ent.label_}>"

        texto_anonimizado = texto_anonimizado[:start] + tag + texto_anonimizado[end:]

    print("\nAnonymized Text:")
    print(texto_anonimizado)


if __name__ == "__main__":
    main()