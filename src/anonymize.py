import argparse
from transformers import pipeline

# Load the NER model
def load_anonymizer(model_path):
    print(f"Loading model from: {model_path}...")
    return pipeline("ner", model=model_path, tokenizer=model_path, aggregation_strategy="simple")


# Detect sensitive entities and replace them with their tags
def anonymize(text, ner_pipeline):
    # Get model predictions
    entities = ner_pipeline(text)

    anonymized_text = text

    # This prevents index shifting when replacing words of different lengths
    for ent in sorted(entities, key=lambda x: x['start'], reverse=True):
        start = ent['start']
        end = ent['end']
        label = ent['entity_group']

        # Format the tag (e.g., from PERSONAL-NAME to [PERSONAL-NAME])
        tag = f"[{label}]"

        # Replace the word with the tag
        anonymized_text = anonymized_text[:start] + tag + anonymized_text[end:]

    return anonymized_text, entities


def main():
    # Setup command line arguments
    parser = argparse.ArgumentParser(description="Anonymize text from municipal minutes.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model folder")
    parser.add_argument("--text", type=str, help="Raw text to anonymize")
    parser.add_argument("--file", type=str, help="Path to a .txt file to anonymize")
    args = parser.parse_args()

    # Check if input is a file or a string
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            input_text = f.read()
    elif args.text:
        input_text = args.text
    else:
        print("Error: You must provide text via --text or a file via --file.")
        return

    # Run the model
    anon_pipeline = load_anonymizer(args.model)
    final_text, detected_entities = anonymize(input_text, anon_pipeline)

    # Print results
    print("\nDETECTED ENTITIES")
    for ent in detected_entities:
        print(f"{ent['word']} -> {ent['entity_group']} (Confidence: {ent['score']:.2f})")

    print("\nANONYMIZED TEXT")
    print(final_text)


if __name__ == "__main__":
    main()