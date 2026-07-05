import os
import sys
import json
import argparse
import numpy as np
import torch
from collections import Counter
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset_processors import create_dataset_processor

# Disable tokenizers parallelism to avoid warnings/deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        loss = self.loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def format_data_for_hf(raw_docs):
    processed_data = []
    allowed_tags = set(label_list)
    for doc in raw_docs:
        corrected_tags = [t if t in allowed_tags else "O" for t in doc['tags']]
        ner_tags_ids = [tag_to_id[tag] for tag in corrected_tags]
        processed_data.append({'tokens': doc['tokens'], 'ner_tags': ner_tags_ids})
    return processed_data


def main():
    # Setup command line arguments
    parser = argparse.ArgumentParser(description="Train NER LOMO Model (Leave-One-Municipality-Out)")
    parser.add_argument("--config", type=str, required=True, help="Config name from JSON")
    parser.add_argument("--target_municipality", type=str, default=None, help="Train for a specific municipality")
    parser.add_argument("--model_name", type=str, help="Override model name")
    parser.add_argument("--data_dir", type=str, help="Override data directory")
    parser.add_argument("--output_dir", type=str, help="Override base output directory")
    args = parser.parse_args()

    # Load configuration
    config_path = "config/training_configs.json"
    with open(config_path, 'r') as f:
        all_configs = json.load(f)

    if args.config not in all_configs:
        print(f"Error: Config '{args.config}' not found")
        return

    params = all_configs[args.config]

    # Overrides
    if args.model_name: params["model_name"] = args.model_name

    # Define paths
    dataset_dir = args.data_dir if args.data_dir else "data/personal_info_dataset"
    base_output_dir = args.output_dir if args.output_dir else f"models/leave_one_municipality_out/{args.config}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {params['model_name']} | Device: {device}")

    # Load Data
    processor = create_dataset_processor('councilseg', dataset_path=dataset_dir)
    all_keys = list(processor.data.keys())

    if args.target_municipality:
        target_prefix = args.target_municipality
        matches = [k for k in all_keys if k.startswith(f"{target_prefix}_")]

        if matches:
            municipalities = [target_prefix]
        else:
            available_prefixes = sorted(list(set([k.split('_')[0] for k in all_keys])))
            return
    else:
        municipalities = sorted(list(set([k.split('_')[0] for k in all_keys])))

    print(f"Starting LOMO for {len(municipalities)} municipalities.")
    tokenizer = AutoTokenizer.from_pretrained(params["model_name"], add_prefix_space=True)

    # Tokenize Function
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, max_length=params["max_length"])
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

    # LOMO Loop
    for test_mun in municipalities:
        print(f"\n--- [LOMO] Leaving out: {test_mun} ---")

        mun_output_dir = os.path.join(base_output_dir, test_mun)
        os.makedirs(mun_output_dir, exist_ok=True)

        other_muns_docs = processor.get_documents(exclude_municipality=test_mun)
        target_mun_docs = processor.get_documents(include_municipality=test_mun)

        if len(other_muns_docs) == 0 or len(target_mun_docs) < 4:
            print(f"Skipping {test_mun}: Insufficient data.")
            continue

        raw_train, raw_val = train_test_split(other_muns_docs, test_size=0.25, random_state=42)
        raw_test = target_mun_docs[:4]

        # Save test docs
        with open(os.path.join(mun_output_dir, "test_docs_info.json"), "w", encoding="utf-8") as f:
            json.dump(raw_test, f, ensure_ascii=False, indent=4)

        # Datasets
        train_dataset = Dataset.from_list(format_data_for_hf(raw_train))
        val_dataset = Dataset.from_list(format_data_for_hf(raw_val))

        tokenized_train = train_dataset.map(tokenize_and_align_labels, batched=True,
                                            remove_columns=train_dataset.column_names)
        tokenized_val = val_dataset.map(tokenize_and_align_labels, batched=True,
                                        remove_columns=val_dataset.column_names)

        # Weights
        all_labels_flat = [l for labels in tokenized_train["labels"] for l in labels if l != -100]
        label_counts = Counter(all_labels_flat)
        class_weights = torch.zeros(len(label_list), dtype=torch.float32).to(device)
        for i in range(len(label_list)):
            count = label_counts.get(i, 0)
            class_weights[i] = np.sqrt(len(all_labels_flat) / count) if count > 0 else 1.0
        class_weights = (class_weights / torch.mean(class_weights)).to(device)

        # Initialize model
        model = AutoModelForTokenClassification.from_pretrained(
            params["model_name"], num_labels=len(label_list), id2label=id_to_tag, label2id=tag_to_id,
            ignore_mismatched_sizes=True
        ).to(device)

        training_params = {k: v for k, v in params.items() if k not in ["model_name", "max_length"]}
        trainer = WeightedLossTrainer(
            model=model, args=TrainingArguments(output_dir=mun_output_dir, **training_params),
            train_dataset=tokenized_train, eval_dataset=tokenized_val, tokenizer=tokenizer,
            data_collator=DataCollatorForTokenClassification(tokenizer), class_weights=class_weights
        )

        print(f"Training started for {test_mun}")
        trainer.train()
        trainer.save_model(mun_output_dir)
        print(f"Model saved in {mun_output_dir}")

        del model, trainer
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()