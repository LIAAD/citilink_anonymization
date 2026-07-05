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

# Few-shot examples used in the prompt (same as those used during experiments)
FEW_SHOT_EXAMPLES = [
    {
        "text": "O Senhor João Silva, residente em Campo Maior, apresentou o processo 1/23.",
        "entities": [
            ("João Silva", "PERSONAL-NAME"),
            ("Campo Maior", "PERSONAL-LOCATION"),
            ("1/23", "PERSONAL-ADMIN")
        ]
    },
    {
        "text": "A Dra. Maria Sousa, residente na Rua das Flores n.º 5, requereu a legalização da fração B.",
        "entities": [
            ("Dra.", "PERSONAL-POSITION"),
            ("Maria Sousa", "PERSONAL-NAME"),
            ("Rua das Flores n.º 5", "PERSONAL-ADDRESS"),
            ("fração B", "PERSONAL-ADMIN")
        ]
    },
    {
        "text": "Não foram identificados requerimentos de particulares nesta reunião.",
        "entities": []
    }
]


def build_prompt(text, few_shot_examples):
    # Build the full instruction prompt with entity types, rules and few-shot examples
    prompt = (
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
        "EXEMPLOS:\n"
    )

    # Append each few-shot example to the prompt
    for ex in few_shot_examples:
        ents_str = str([(e[0], e[1]) for e in ex["entities"]]) if ex["entities"] else "[]"
        prompt += f'Texto: "{ex["text"]}"\nEntidades: {ents_str}\n\n'

    # Append the actual document text to predict
    prompt += f'Texto: "{text}"\nEntidades:'
    return prompt


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


def call_gemini(prompt, params):
    # Call the Gemini API using the google-generativeai library
    import google.generativeai as genai

    # Read the API key from the environment variable defined in the config
    api_key = os.environ.get(params["api_key_env"])
    if not api_key:
        raise ValueError(f"Environment variable '{params['api_key_env']}' is not set.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(params["model_name"])

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=params.get("temperature", 0.0),
            max_output_tokens=params.get("max_output_tokens", 8192)
        )
    )
    return response.text


def call_local_llm(prompt, params):
    # Call a local causal LM (GerVASIO or AMALIA) via HuggingFace transformers with optional 4-bit quantisation
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch

    # Configure 4-bit quantisation to reduce GPU memory usage
    quantization_config = BitsAndBytesConfig(load_in_4bit=params.get("load_in_4bit", True))
    tokenizer = AutoTokenizer.from_pretrained(
        params["model_name"],
        local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        params["model_name"],
        quantization_config=quantization_config,
        device_map="auto",
        local_files_only=True
    )

    # Format the input as a chat message for instruction-tuned models
    messages  = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

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
    # Sort by descending position to avoid index shifts when replacing
    sorted_ents = sorted(entities, key=lambda e: text.find(e["text"]), reverse=True)
    pseudonymized = text
    for ent in sorted_ents:
        span  = ent["text"]
        label = ent["label"].replace("PERSONAL-", "")
        tag   = f"<{label}>"
        # Replace only the first occurrence to avoid over-substitution
        pseudonymized = pseudonymized.replace(span, tag, 1)
    return pseudonymized


def main():
    # Setup command line arguments
    parser = argparse.ArgumentParser(description="Run few-shot NER pipeline on all splits")
    parser.add_argument("--config", type=str, required=True,
                        help="Config name (e.g., gemini_2_5_pro, gervasio_8b or amalia_9b)")
    parser.add_argument("--data_dir", type=str, default="data/personal_info_dataset",
                        help="Path to the dataset directory")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to run on: train, val or test (default: test)")
    args = parser.parse_args()

    # Load configuration from JSON
    config_path = "config/training_configs.json"
    with open(config_path, "r") as f:
        all_configs = json.load(f)

    if args.config not in all_configs:
        print(f"Error: Config '{args.config}' not found in {config_path}")
        return

    params      = all_configs[args.config]
    output_dir  = f"results/generative/few_shot/{args.config}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Running few-shot pipeline: {params['model_name']} on split '{args.split}'")

    # Load documents from the requested split
    processor = create_dataset_processor("councilseg", dataset_path=args.data_dir)
    docs      = processor.get_documents(split=args.split)
    print(f"Documents loaded: {len(docs)}")

    # Select the number of few-shot examples defined in the config
    n_shots       = params.get("few_shot_examples", 2)
    shot_examples = FEW_SHOT_EXAMPLES[:n_shots]

    pipeline_outputs = []

    for i, doc in enumerate(docs):
        # Reconstruct plain text from the token list
        text = " ".join(doc["tokens"])

        # Build the few-shot prompt for this document
        prompt = build_prompt(text, shot_examples)

        # Route to the correct model backend based on the config name
        if args.config == "gemini_2_5_pro":
            raw_output = call_gemini(prompt, params)
        elif args.config in ("gervasio_8b", "amalia_9b"):
            raw_output = call_local_llm(prompt, params)
        else:
            print(f"Error: Unknown config '{args.config}'. Supported: gemini_2_5_pro, gervasio_8b, amalia_9b")
            return

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
    output_path = os.path.join(output_dir, f"pipeline_outputs_{args.split}_{args.config}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_outputs, f, indent=4, ensure_ascii=False)

    print(f"Pipeline complete. Results saved in: {output_path}")


if __name__ == "__main__":
    main()