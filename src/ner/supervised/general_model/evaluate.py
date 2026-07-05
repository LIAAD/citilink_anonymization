import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    DataCollatorForTokenClassification
)
from seqeval.metrics import classification_report as seq_classification_report
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset_processors import create_dataset_processor
from sklearn.metrics import confusion_matrix

# Define label mappings
label_list = [
    "O", "B-PERSONAL-NAME", "I-PERSONAL-NAME", "B-PERSONAL-ADMIN", "I-PERSONAL-ADMIN",
    "B-PERSONAL-POSITION", "I-PERSONAL-POSITION", "B-PERSONAL-ADDRESS", "I-PERSONAL-ADDRESS",
    "B-PERSONAL-DATE", "I-PERSONAL-DATE", "B-PERSONAL-LOCATION", "I-PERSONAL-LOCATION",
    "B-PERSONAL-OTHER", "I-PERSONAL-OTHER", "B-PERSONAL-INFO", "I-PERSONAL-INFO",
    "B-PERSONAL-COMPANY", "I-PERSONAL-COMPANY", "B-PERSONAL-ARTISTIC", "I-PERSONAL-ARTISTIC",
    "B-PERSONAL-DEGREE", "I-PERSONAL-DEGREE", "B-PERSONAL-TIME", "I-PERSONAL-TIME",
    "B-PERSONAL-LICENSE", "I-PERSONAL-LICENSE", "B-PERSONAL-JOB", "I-PERSONAL-JOB",
    "B-PERSONAL-VEHICLE", "I-PERSONAL-VEHICLE", "B-PERSONAL-FACULTY", "I-PERSONAL-FACULTY",
    "B-PERSONAL-FAMILY", "I-PERSONAL-FAMILY"
]

tag_to_id = {tag: i for i, tag in enumerate(label_list)}
id_to_tag = {i: tag for i, tag in enumerate(label_list)}


# Function to align labels with subword tokens
def tokenize_and_align_labels(examples, tokenizer, max_length):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, max_length=max_length)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def main():
    # Setup command line arguments
    parser = argparse.ArgumentParser(description="Evaluate NER model on Test Set")
    parser.add_argument("--config", type=str, required=True, help="Config name (e.g., xlm_roberta or bert_base)")
    parser.add_argument("--model_path", type=str, help="Path to the trained model (overrides default)")
    args = parser.parse_args()

    # Load configuration
    config_path = "config/training_configs.json"
    with open(config_path, 'r') as f:
        all_configs = json.load(f)

    if args.config not in all_configs:
        print(f"Error: Config '{args.config}' not found in {config_path}")
        return

    params = all_configs[args.config]

    # Define paths
    dataset_dir = "data/personal_info_dataset"

    # If model_path is not provided, assume the default Models/ folder
    model_dir = args.model_path if args.model_path else f"models/general_model/{args.config}"
    results_dir = f"src/results/general_model/{args.config}"

    os.makedirs(results_dir, exist_ok=True)

    print(f"Evaluating {args.config}")
    print(f"Loading model from: {model_dir}")

    # Load Test Data using the Processor
    processor = create_dataset_processor('councilseg', dataset_path=dataset_dir)

    raw_test_docs = processor.get_documents(split='test')

    # Convert text tags to numeric IDs and filter out unknown tags
    processed_test_data = []
    allowed_tags = set(label_list)
    for doc in raw_test_docs:
        corrected_tags = [t if t in allowed_tags else "O" for t in doc['tags']]
        ner_tags_ids = [tag_to_id[tag] for tag in corrected_tags]
        processed_test_data.append({
            'tokens': doc['tokens'],
            'ner_tags': ner_tags_ids
        })

    test_dataset = Dataset.from_list(processed_test_data)
    print(f"Test documents loaded: {len(test_dataset)}")

    # Load Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(model_dir, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)

    # Tokenize Data
    tokenized_test = test_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, params["max_length"]),
        batched=True,
        remove_columns=test_dataset.column_names
    )

    # Initialize Trainer (only for prediction)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # Get Predictions
    print("Extracting predictions on test set...")
    predictions_output = trainer.predict(tokenized_test)
    predictions = np.argmax(predictions_output.predictions, axis=2)
    labels = predictions_output.label_ids

    # Convert numeric IDs back to BIO tags, ignoring -100 (padding/subwords)
    true_labels_bio = [[id_to_tag[l] for l in label if l != -100] for label in labels]
    pred_labels_bio = [[id_to_tag[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in
                       zip(predictions, labels)]

    # Calculate Metrics (seqeval)
    report = seq_classification_report(true_labels_bio, pred_labels_bio, output_dict=True, mode='strict',
                                       zero_division=0)

    # Clean nested numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        elif hasattr(obj, 'item'):
            return obj.item()
        return obj

    clean_report = convert_numpy(report)

    metrics_path = os.path.join(results_dir, f"metrics_{args.config}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(clean_report, f, indent=4, ensure_ascii=False)

    # Generate Confusion Matrix
    print("Create Confusion Matrix")
    true_flat = [tag.split('-', 1)[-1] if '-' in tag else tag for sublist in true_labels_bio for tag in sublist]
    pred_flat = [tag.split('-', 1)[-1] if '-' in tag else tag for sublist in pred_labels_bio for tag in sublist]
    unique_labels = sorted(list(set(true_flat + pred_flat)))
    if 'O' in unique_labels:
        unique_labels.remove('O')
        unique_labels.insert(0, 'O')

    cm = confusion_matrix(true_flat, pred_flat, labels=unique_labels)
    plt.figure(figsize=(14, 12))
    sns.heatmap(pd.DataFrame(cm, index=unique_labels, columns=unique_labels), annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {args.config.upper()}")

    cm_path = os.path.join(results_dir, f"confusion_matrix_{args.config}.png")
    plt.savefig(cm_path)
    plt.close()

    print(f"Evaluation complete, Results saved in: {results_dir}")


if __name__ == '__main__':
    main()