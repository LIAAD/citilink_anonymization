import os
import sys
import json
import argparse
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from dataset_processors import create_dataset_processor

# Define all valid entity labels
VALID_LABELS = [
    "PERSONAL-NAME", "PERSONAL-ADMIN", "PERSONAL-POSITION", "PERSONAL-ADDRESS",
    "PERSONAL-DATE", "PERSONAL-LOCATION", "PERSONAL-OTHER", "PERSONAL-INFO",
    "PERSONAL-COMPANY", "PERSONAL-ARTISTIC", "PERSONAL-DEGREE", "PERSONAL-TIME",
    "PERSONAL-LICENSE", "PERSONAL-JOB", "PERSONAL-VEHICLE", "PERSONAL-FACULTY",
    "PERSONAL-FAMILY"
]

# Inference prompt — no few-shot examples, model relies on fine-tuning
INSTRUCTION = (
    "És um sistema especializado no reconhecimento de informação pessoal (NER) aplicado a atas de "
    "reuniões de câmaras municipais portuguesas. O teu único objetivo é extrair dados pessoais, ou seja, "
    "qualquer informação que permita identificar, de forma direta ou indireta, uma pessoa específica.\n\n"
    "TIPOS DE ENTIDADE E EXEMPLOS:\n"
    "- PERSONAL-NAME: nome próprio completo de uma pessoa (ex.: 'Ana Cristina Silva Rodrigues')\n"
    "- PERSONAL-ADMIN: referência administrativa pessoal (ex.: '1/23-LEGALIZACAO')\n"
    "- PERSONAL-POSITION: cargo associado a uma pessoa identificada (ex.: 'Chefe de Gabinete')\n"
    "- PERSONAL-ADDRESS: morada ou residência de uma pessoa (ex.: 'Lote 24 do Loteamento das Caraças')\n"
    "- PERSONAL-LOCATION: localização associada a uma pessoa (ex.: 'Campo Maior')\n"
    "- PERSONAL-DATE: data associada a uma pessoa (ex.: '15 de março de 1980')\n"
    "- PERSONAL-INFO: informação pessoal ou biográfica (ex.: '109769031')\n"
    "- PERSONAL-COMPANY: empresa associada a uma pessoa (ex.: 'Silvares, Unipessoal, Lda.')\n"
    "- PERSONAL-ARTISTIC: nome artístico ou pseudónimo (ex.: 'DJ Sunny')\n"
    "- PERSONAL-DEGREE: grau ou título académico (ex.: 'Licenciatura em Direito')\n"
    "- PERSONAL-TIME: hora associada a uma pessoa (ex.: '14h30')\n"
    "- PERSONAL-LICENSE: matrícula ou número de registo (ex.: 'AA-12-BB')\n"
    "- PERSONAL-JOB: profissão ou ocupação (ex.: 'eletricista')\n"
    "- PERSONAL-VEHICLE: identificação de um veículo (ex.: 'Renault Clio')\n"
    "- PERSONAL-FACULTY: instituição de ensino superior (ex.: 'Faculdade de Letras')\n"
    "- PERSONAL-FAMILY: relação de parentesco (ex.: 'filho de')\n"
    "- PERSONAL-OTHER: outra informação pessoal (ex.: 'défice cognitivo')\n\n"
    "REGRAS:\n"
    "- Copia o excerto EXATAMENTE como aparece no texto, sem o alterar.\n"
    "- NUNCA extrair informação pública, institucional ou genérica.\n"
    "- Cada entidade deve poder ser associada a uma pessoa específica.\n"
    "- Se não existir nenhuma entidade pessoal, devolver uma lista vazia: []\n"
    "- Não inventar, extrair apenas o que está literalmente no texto.\n\n"
)


def build_prompt(text):
    # Build the inference prompt — no few-shot examples since the model was fine-tuned
    return INSTRUCTION + f'Texto: "{text}"\nEntidades:'


def parse_model_output(raw_output):
    # Parse entity tuples from the model's raw string output
    # Expected format: [("text", "LABEL"), ...]
    entities = []
    pattern  = r'\(\s*["\'](.+?)["\'],\s*["\']?(PERSONAL-[A-Z]+)["\']?\s*\)'
    matches  = re.findall(pattern, raw_output)
    for text, label in matches:
        if label in VALID_LABELS:
            entities.append({"text": text.strip(), "label": label})
    return entities


def compute_metrics(true_entities_list, pred_entities_list):
    # Compute precision, recall and F1 using exact match on (text, label) pairs
    tp = fp = fn = 0
    for true_ents, pred_ents in zip(true_entities_list, pred_entities_list):
        true_set = {(e["text"], e["label"]) for e in true_ents}
        pred_set = {(e["text"], e["label"]) for e in pred_ents}
        tp += len(true_set & pred_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "tp": tp, "fp": fp, "fn": fn
    }


def load_model(model_path, params):
    # Load the fine-tuned model and tokenizer from a local directory
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch

    # Configure 4-bit quantisation to reduce GPU memory usage during inference
    quantization_config = BitsAndBytesConfig(load_in_4bit=params.get("load_in_4bit", True))
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model     = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        local_files_only=True
    )
    model.eval()
    return model, tokenizer


def run_inference(prompt, model, tokenizer, params):
    # Run inference on a single prompt and return the decoded output string
    import torch

    # Format the input as a chat message for instruction-tuned models
    messages  = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=params.get("max_new_tokens", 1024),
            do_sample=params.get("do_sample", False),
            temperature=params.get("temperature", 0.0)
        )

    # Decode only the newly generated tokens, excluding the input prompt
    generated = output[0][input_ids.shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    # Setup command line arguments
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned generative NER model on test set")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the fine-tuned model directory (GerVASIO or AMALIA)")
    parser.add_argument("--config",     type=str, default="gervasio_8b",
                        help="Config name: gervasio_8b or amalia_9b")
    parser.add_argument("--data_dir", type=str, default="data/personal_info_dataset",
                        help="Path to the dataset directory")
    args = parser.parse_args()

    # Load configuration from JSON to retrieve inference parameters
    config_path = "config/training_configs.json"
    with open(config_path, "r") as f:
        all_configs = json.load(f)

    params      = all_configs.get(args.config, all_configs.get("gervasio_8b", {}))
    results_dir = f"results/generative/fine_tuned/{args.config}"
    os.makedirs(results_dir, exist_ok=True)

    print(f"Evaluating fine-tuned model ({args.config}) from: {args.model_path}")

    # Load the fine-tuned model and tokenizer
    model, tokenizer = load_model(args.model_path, params)

    # Load test documents via the shared dataset processor
    processor = create_dataset_processor("councilseg", dataset_path=args.data_dir)
    test_docs  = processor.get_documents(split="test")
    print(f"Test documents loaded: {len(test_docs)}")

    true_entities_list = []
    pred_entities_list = []
    raw_outputs        = []

    for i, doc in enumerate(test_docs):
        # Reconstruct plain text from the token list
        text = " ".join(doc["tokens"])

        # Build the inference prompt without few-shot examples
        prompt = build_prompt(text)

        # Run inference with the fine-tuned model
        raw_output = run_inference(prompt, model, tokenizer, params)

        # Parse the model's raw string output into structured entity dicts
        pred_ents = parse_model_output(raw_output)

        # Convert gold BIO tags to entity spans for comparison
        true_ents = []
        tokens    = doc["tokens"]
        tags      = doc["tags"]
        j = 0
        while j < len(tags):
            if tags[j].startswith("B-"):
                label      = tags[j][2:]
                ent_tokens = [tokens[j]]
                k = j + 1
                # Collect all consecutive I- tokens belonging to this entity
                while k < len(tags) and tags[k] == f"I-{label}":
                    ent_tokens.append(tokens[k])
                    k += 1
                true_ents.append({"text": " ".join(ent_tokens), "label": label})
                j = k
            else:
                j += 1

        true_entities_list.append(true_ents)
        pred_entities_list.append(pred_ents)
        raw_outputs.append({"doc_index": i, "raw_output": raw_output, "parsed": pred_ents})

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(test_docs)} documents")

    # Compute overall precision, recall and F1
    metrics = compute_metrics(true_entities_list, pred_entities_list)
    print(f"\nResults for fine-tuned {args.config}:")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")

    # Save metrics to JSON
    metrics_path = os.path.join(results_dir, f"metrics_{args.config}_finetuned.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    # Save raw model outputs for error analysis
    raw_path = os.path.join(results_dir, f"raw_outputs_{args.config}_finetuned.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_outputs, f, indent=4, ensure_ascii=False)

    print(f"Evaluation complete. Results saved in: {results_dir}")


if __name__ == "__main__":
    main()