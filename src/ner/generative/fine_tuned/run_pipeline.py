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


def load_model(model_path, params):
    # Load the fine-tuned model and tokenizer from a local directory
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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


def pseudonymize(text, entities):
    # Replace detected entity spans in the text with pseudonym tags
    # Sort in descending order of position to avoid index shifts when replacing
    pseudonymized = text
    for ent in sorted(entities, key=lambda e: text.find(e["text"]), reverse=True):
        span  = ent["text"]
        label = ent["label"].replace("PERSONAL-", "")
        tag   = f"<{label}>"
        # Replace only the first occurrence to avoid over-substitution
        pseudonymized = pseudonymized.replace(span, tag, 1)
    return pseudonymized


def main():
    # Setup command line arguments
    parser = argparse.ArgumentParser(description="Run fine-tuned generative NER pipeline")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the fine-tuned model directory (GerVASIO or AMALIA)")
    parser.add_argument("--config",     type=str, default="gervasio_8b",
                        help="Config name: gervasio_8b or amalia_9b")
    parser.add_argument("--data_dir", type=str, default="data/personal_info_dataset",
                        help="Path to the dataset directory")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to run on: train, val or test (default: test)")
    args = parser.parse_args()

    # Load configuration from JSON to retrieve inference parameters
    config_path = "config/training_configs.json"
    with open(config_path, "r") as f:
        all_configs = json.load(f)

    params     = all_configs.get(args.config, all_configs.get("gervasio_8b", {}))
    output_dir = f"results/generative/fine_tuned/{args.config}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Running fine-tuned {args.config} pipeline on split '{args.split}'")
    print(f"Model path: {args.model_path}")

    # Load the fine-tuned model and tokenizer from the provided directory
    model, tokenizer = load_model(args.model_path, params)

    # Load documents from the requested split
    processor = create_dataset_processor("councilseg", dataset_path=args.data_dir)
    docs      = processor.get_documents(split=args.split)
    print(f"Documents loaded: {len(docs)}")

    pipeline_outputs = []

    for i, doc in enumerate(docs):
        # Reconstruct plain text from the token list
        text = " ".join(doc["tokens"])

        # Build the inference prompt without few-shot examples
        prompt = build_prompt(text)

        # Run inference with the fine-tuned model
        raw_output = run_inference(prompt, model, tokenizer, params)

        # Parse the model's raw string output into structured entity dicts
        pred_entities = parse_model_output(raw_output)

        # Pseudonymize the original text using the predicted entities
        pseudonymized_text = pseudonymize(text, pred_entities)

        pipeline_outputs.append({
            "doc_key":            doc.get("doc_key", f"doc_{i}"),
            "original_text":      text,
            "pseudonymized_text": pseudonymized_text,
            "entities":           pred_entities,
            "raw_output":         raw_output
        })

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(docs)} documents")

    # Save all pipeline outputs to a single JSON file
    output_path = os.path.join(
        output_dir, f"pipeline_outputs_{args.split}_{args.config}_finetuned.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_outputs, f, indent=4, ensure_ascii=False)

    print(f"Pipeline complete. Results saved in: {output_path}")


if __name__ == "__main__":
    main()