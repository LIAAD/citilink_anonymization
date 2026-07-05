import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

import torch
import torch.nn as nn
from transformers import pipeline, AutoTokenizer, AutoModel

try:
    from tqdm import tqdm
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm", "-q"])
    from tqdm import tqdm

DIST_BUCKETS      = [1, 2, 3, 4, 5, 8, 16, 32, 64]
MIN_CLUSTER_SCORE = 0.3
NER_UNION_MARGIN  = 0.5


# NER
def load_ner_model(model_path):
    # Load the NER model as a HuggingFace pipeline
    print(f"Loading NER model from: {model_path}...")
    return pipeline(
        "ner", model=model_path, tokenizer=model_path,
        aggregation_strategy="simple"
    )


def run_ner(text: str, ner_pipeline) -> List[Dict]:
    # Run NER and return list of detected entities with char offsets
    return ner_pipeline(text)


# Coreference Model Architecture
class CorefModel(nn.Module):
    def __init__(self, model_dir, ffnn_dim, max_span_width,
                 max_antecedents, max_cluster_size, dropout=0.0):
        super().__init__()
        self.encoder          = AutoModel.from_pretrained(str(model_dir))
        H                     = self.encoder.config.hidden_size
        self.hidden           = H
        self.max_span_width   = max_span_width
        self.max_antecedents  = max_antecedents
        self.max_cluster_size = max_cluster_size

        self.width_emb = nn.Embedding(max_span_width + 1, 20)
        self.head_attn = nn.Linear(H, 1)
        span_dim       = H * 3 + 20

        self.pos_emb        = nn.Embedding(20, 32)
        self.mention_scorer = nn.Sequential(
            nn.Linear(span_dim + 32, ffnn_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(ffnn_dim, ffnn_dim),       nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(ffnn_dim, 1),
        )

        n_dist        = len(DIST_BUCKETS) + 1
        self.dist_emb = nn.Embedding(n_dist, 20)
        self.register_buffer(
            "dist_bucket_boundaries",
            torch.tensor(DIST_BUCKETS, dtype=torch.long)
        )
        ant_dim = span_dim * 3 + 20
        self.ant_scorer = nn.Sequential(
            nn.Linear(ant_dim, ffnn_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(ffnn_dim, ffnn_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(ffnn_dim, 1),
        )
        self.epsilon = nn.Parameter(torch.zeros(1))

    def encode_window(self, input_ids, attention_mask, device):
        return self.encoder(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
        ).last_hidden_state

    def span_repr(self, doc_h, starts, ends):
        n      = len(starts)
        H      = doc_h.shape[1]
        h_s    = doc_h[starts]
        h_e    = doc_h[ends]
        widths = (ends - starts).clamp(0, self.max_span_width)
        w_emb  = self.width_emb(widths)
        W        = int(widths.max().item()) + 1
        span_tok = torch.zeros(n, W, H, device=doc_h.device, dtype=doc_h.dtype)
        mask     = torch.zeros(n, W, dtype=torch.bool, device=doc_h.device)
        for i in range(n):
            s = starts[i].item()
            w = int(widths[i].item()) + 1
            span_tok[i, :w] = doc_h[s:s + w]
            mask[i, :w]     = True
        attn = self.head_attn(span_tok)
        attn[~mask.unsqueeze(-1).expand_as(attn)] = -1e9
        attn = torch.softmax(attn, dim=1)
        head = (attn * span_tok).sum(dim=1)
        return torch.cat([h_s, h_e, head, w_emb], dim=-1)

    def mention_score_with_pos(self, span_vecs, starts, doc_len):
        pos_ratio  = starts.float() / max(doc_len - 1, 1)
        pos_bucket = (pos_ratio * 19).long().clamp(0, 19)
        pos_emb    = self.pos_emb(pos_bucket)
        return self.mention_scorer(
            torch.cat([span_vecs, pos_emb], dim=-1)
        ).squeeze(-1)

    def _pair_vec(self, span_vecs, i, js):
        if not isinstance(js, torch.Tensor):
            js = torch.tensor(js, dtype=torch.long, device=span_vecs.device)
        vi    = span_vecs[i].unsqueeze(0).expand(len(js), -1)
        vj    = span_vecs[js]
        dist  = (i - js).clamp(min=0)
        d_bkt = torch.bucketize(dist, self.dist_bucket_boundaries)
        d_emb = self.dist_emb(d_bkt)
        return torch.cat([vi, vj, vi * vj, d_emb], dim=-1)

# Coreference Loading & Inference
def load_coref_model(model_path: str, device):
    # Load the coreference model from a local directory with coref_config.json
    model_dir = Path(model_path)
    print(f"Loading coreference model from: {model_dir}...")

    with open(model_dir / "coref_config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    model = CorefModel(
        model_dir=str(model_dir),
        ffnn_dim=cfg["ffnn_dim"],
        max_span_width=cfg["max_span_width"],
        max_antecedents=cfg["max_antecedents"],
        max_cluster_size=cfg["max_cluster_size"],
        dropout=0.0,
    )

    cabecas     = torch.load(model_dir / "cabecas_coref.pt",
                             map_location=device, weights_only=False)
    model_state = model.state_dict()
    model_state.update({k: v for k, v in cabecas.items() if k in model_state})
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    tokenizer  = AutoTokenizer.from_pretrained(str(model_dir))
    max_tokens = cfg.get("max_tokens", 512)
    stride     = cfg.get("stride", 128)
    epsilon    = cfg.get("epsilon", model.epsilon.item())

    return model, tokenizer, max_tokens, stride, epsilon


def sliding_windows(tokens, tokenizer, max_tokens, stride):
    flat = []
    for idx, tok in enumerate(tokens):
        subs = tokenizer.encode(tok, add_special_tokens=False) or [tokenizer.unk_token_id]
        flat.extend((idx, s) for s in subs)

    cls_id, sep_id, pad_id = (tokenizer.cls_token_id,
                               tokenizer.sep_token_id,
                               tokenizer.pad_token_id)
    max_content = max_tokens - 2
    windows, start = [], 0
    while start < len(flat):
        end   = min(start + max_content, len(flat))
        chunk = flat[start:end]
        ids   = [cls_id] + [s for _, s in chunk] + [sep_id]
        offs  = [-1]     + [o for o, _ in chunk]  + [-1]
        mask  = [1] * len(ids)
        pad   = max_tokens - len(ids)
        ids  += [pad_id] * pad; mask += [0] * pad; offs += [-1] * pad
        windows.append({"input_ids": ids, "attention_mask": mask, "token_offsets": offs})
        if end == len(flat):
            break
        start += max_content - stride
    return windows


def aggregate_embeddings(hidden_list, offsets_list, n_tok, H, device):
    all_h    = torch.cat([wh.reshape(-1, H) for wh in hidden_list], dim=0)
    all_offs = []
    for offs in offsets_list:
        all_offs.extend(offs)
    all_offs = torch.tensor(all_offs, dtype=torch.long, device=device)
    valid    = all_offs >= 0
    h_val    = all_h[valid]
    t_val    = all_offs[valid].clamp(0, n_tok - 1)
    acc      = torch.zeros(n_tok, H, device=device, dtype=all_h.dtype)
    count    = torch.zeros(n_tok,    device=device, dtype=all_h.dtype)
    acc.scatter_add_(0, t_val.unsqueeze(1).expand(-1, H), h_val)
    count.scatter_add_(0, t_val,
                       torch.ones(len(t_val), device=device, dtype=all_h.dtype))
    return acc / count.clamp(min=1.0).unsqueeze(-1)


@torch.no_grad()
def predict_coref_clusters(model, tokenizer, tokens, ner_spans,
                            max_tokens, stride, epsilon, device):
    # Run coreference resolution on token-level NER spans
    # ner_spans: list of (start_tok, end_tok, label)
    if not ner_spans:
        return []

    all_spans     = sorted({(s, e) for s, e, _ in ner_spans})
    span_to_label = {(s, e): lbl for s, e, lbl in ner_spans}
    windows       = sliding_windows(tokens, tokenizer, max_tokens, stride)
    if not windows:
        return []

    hidden_list, offsets_list = [], []
    for w in windows:
        inp  = torch.tensor([w["input_ids"]],      dtype=torch.long)
        mask = torch.tensor([w["attention_mask"]], dtype=torch.long)
        hidden_list.append(model.encode_window(inp, mask, device)[0])
        offsets_list.append(w["token_offsets"])

    max_tok = len(tokens)
    doc_h   = aggregate_embeddings(
        hidden_list, offsets_list, max_tok, model.hidden, device
    )

    ss = torch.tensor([s for s, e in all_spans], dtype=torch.long,
                      device=device).clamp(0, max_tok - 1)
    es = torch.tensor([e for s, e in all_spans], dtype=torch.long,
                      device=device).clamp(0, max_tok - 1)
    sv = model.span_repr(doc_h, ss, es)
    ms = model.mention_score_with_pos(sv, ss, max_tok)

    parent  = list(range(len(all_spans)))
    cl_size = [1] * len(all_spans)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb: return
        if cl_size[ra] + cl_size[rb] > model.max_cluster_size: return
        parent[ra]   = rb
        cl_size[rb] += cl_size[ra]

    threshold     = max(epsilon, MIN_CLUSTER_SCORE)
    ner_threshold = threshold + NER_UNION_MARGIN

    for i in range(1, len(all_spans)):
        j_max = min(i, model.max_antecedents)
        js    = list(range(i - j_max, i))
        if not js: continue
        js_t   = torch.tensor(js, dtype=torch.long, device=device)
        pair   = model._pair_vec(sv, i, js_t)
        scores = model.ant_scorer(pair).squeeze(-1) + ms[i] + ms[js_t]
        best_l = scores.argmax().item()
        best_s = scores[best_l].item()
        best_j = js[best_l]
        if best_s > ner_threshold:
            si, sj = all_spans[i], all_spans[best_j]
            if span_to_label.get(si) == span_to_label.get(sj):
                union(i, best_j)

    clusters_out: Dict[int, list] = defaultdict(list)
    for i in range(len(all_spans)):
        clusters_out[find(i)].append(all_spans[i])
    return [v for v in clusters_out.values() if len(v) >= 2]

# Conversion: char spans ↔ token spans
def build_char_to_token_map(tokens: List[str]) -> Dict[int, int]:
    # Build a mapping from character position to token index
    char_to_tok = {}
    pos = 0
    for tok_idx, tok in enumerate(tokens):
        for c in range(pos, pos + len(tok)):
            char_to_tok[c] = tok_idx
        pos += len(tok) + 1
    return char_to_tok


def char_spans_to_token_spans(ner_entities, char_to_tok, tokens, text):
    # Convert NER char-level spans to token-level spans
    result = []
    for ent in ner_entities:
        sc    = ent["start"]
        ec    = ent["end"]
        label = ent["entity_group"]
        s_tok = char_to_tok.get(sc)
        e_tok = char_to_tok.get(min(ec - 1, len(text) - 1))
        if s_tok is not None and e_tok is not None:
            result.append((s_tok, e_tok, label))
    return result

# Pseudonymization
def assign_pseudonym_ids(ner_entities, clusters, char_to_tok, text):
    # Build a mapping from char_start → cluster_id using coreference clusters
    span_to_cluster_id: Dict[int, int] = {}
    cluster_counters: Dict[str, int]   = defaultdict(int)

    for cluster in clusters:
        # Determine the label of this cluster from the first entity found
        cluster_label = None
        cluster_char_starts = []
        for (s_tok, e_tok) in cluster:
            # Find the entity whose token span matches
            for ent in ner_entities:
                st = char_to_tok.get(ent["start"])
                et = char_to_tok.get(min(ent["end"] - 1, len(text) - 1))
                if st == s_tok and et == e_tok:
                    cluster_char_starts.append(ent["start"])
                    if cluster_label is None:
                        cluster_label = ent["entity_group"]
                    break
        if not cluster_label or not cluster_char_starts:
            continue
        cluster_counters[cluster_label] += 1
        cid = cluster_counters[cluster_label]
        for char_s in cluster_char_starts:
            span_to_cluster_id[char_s] = cid

    # Assign IDs: cluster ID if coreferent, new sequential ID otherwise
    entity_counters: Dict[str, int]        = defaultdict(int)
    text_to_id: Dict[Tuple[str, str], int] = {}
    result = []

    for ent in sorted(ner_entities, key=lambda e: e["start"]):
        label     = ent["entity_group"]
        char_s    = ent["start"]
        text_key  = ent["word"].lower()

        if char_s in span_to_cluster_id:
            assigned_id = span_to_cluster_id[char_s]
        elif (label, text_key) in text_to_id:
            assigned_id = text_to_id[(label, text_key)]
        else:
            entity_counters[label] += 1
            assigned_id = entity_counters[label]
            text_to_id[(label, text_key)] = assigned_id

        result.append({**ent, "pseudonym_id": assigned_id})

    return result


def replace_entities(text: str, entities_with_ids: List[Dict]) -> str:
    # Replace each detected entity with a pseudonym tag including its ID
    pseudonymized = text
    for ent in sorted(entities_with_ids, key=lambda e: e["start"], reverse=True):
        start = ent["start"]
        end   = ent["end"]
        label = ent["entity_group"]
        pid   = ent["pseudonym_id"]
        tag   = f"<{label}-{pid}>"
        pseudonymized = pseudonymized[:start] + tag + pseudonymized[end:]
    return re.sub(r' +', ' ', pseudonymized).strip()


# Full Pipeline
def pseudonymize(text: str, ner_pipeline, coref_model, coref_tokenizer,
                 max_tokens, stride, epsilon, device) -> Tuple[str, List[Dict]]:
    # Step 1: Run NER to detect sensitive entities
    ner_entities = run_ner(text, ner_pipeline)
    if not ner_entities:
        return text, []

    # Step 2: Tokenize text and build char→token mapping
    tokens       = text.split()
    char_to_tok  = build_char_to_token_map(tokens)
    ner_tok_spans = char_spans_to_token_spans(ner_entities, char_to_tok, tokens, text)

    # Step 3: Run coreference resolution on NER spans
    clusters = predict_coref_clusters(
        coref_model, coref_tokenizer, tokens, ner_tok_spans,
        max_tokens, stride, epsilon, device
    )

    # Step 4: Assign consistent pseudonym IDs using coreference clusters
    entities_with_ids = assign_pseudonym_ids(ner_entities, clusters, char_to_tok, text)

    # Step 5: Replace entities in text with pseudonym tags
    pseudonymized_text = replace_entities(text, entities_with_ids)

    return pseudonymized_text, entities_with_ids

# Main
def main():
    parser = argparse.ArgumentParser(
        description="Pseudonymize text from Portuguese municipal minutes "
                    "using NER + Coreference Resolution."
    )
    parser.add_argument("--ner_model",   type=str, required=True,
                        help="Path to the trained NER model folder")
    parser.add_argument("--coref_model", type=str, required=True,
                        help="Path to the trained coreference model folder")
    parser.add_argument("--text", type=str,
                        help="Raw text to pseudonymize")
    parser.add_argument("--file", type=str,
                        help="Path to a .txt file to pseudonymize")
    args = parser.parse_args()

    # Read input
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            input_text = f.read()
    elif args.text:
        input_text = args.text
    else:
        print("Error: provide text via --text or a file via --file.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load models
    ner_pipeline = load_ner_model(args.ner_model)
    coref_model, coref_tokenizer, max_tokens, stride, epsilon = load_coref_model(
        args.coref_model, device
    )

    # Run pseudonymization pipeline
    pseudonymized_text, entities = pseudonymize(
        input_text, ner_pipeline,
        coref_model, coref_tokenizer,
        max_tokens, stride, epsilon, device
    )

    # Print results
    print("\nDETECTED ENTITIES")
    print(f"  {'Entity':<40} {'Label':<25} {'ID':>4} {'Confidence':>10}")
    print(f"  {'-'*83}")
    for ent in sorted(entities, key=lambda e: e["start"]):
        print(f"  {ent['word']:<40} {ent['entity_group']:<25} "
              f"{ent['pseudonym_id']:>4} {ent['score']:>10.4f}")

    print("\nPSEUDONYMIZED TEXT")
    print(pseudonymized_text)


if __name__ == "__main__":
    main()