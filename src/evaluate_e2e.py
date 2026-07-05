import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification

try:
    from tqdm import tqdm
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm", "-q"])
    from tqdm import tqdm

VALID_NER_CLASSES = {
    "PERSONAL-NAME", "PERSONAL-ADMIN", "PERSONAL-POSITION", "PERSONAL-LOCATION",
    "PERSONAL-ADDRESS", "PERSONAL-DATE", "PERSONAL-OTHER", "PERSONAL-COMPANY",
    "PERSONAL-INFO", "PERSONAL-ARTISTIC", "PERSONAL-JOB", "PERSONAL-DEGREE",
    "PERSONAL-TIME", "PERSONAL-FACULTY", "PERSONAL-FAMILY", "PERSONAL-LICENSE",
    "PERSONAL-VEHICLE",
}

# Maps NER model output labels (PascalCase) to gold standard labels (UPPERCASE)
NER_LABEL_TO_GOLD_CLASS = {
    "PERSONAL-Name":                      "PERSONAL-NAME",
    "PERSONAL-AdministrativeInformation": "PERSONAL-ADMIN",
    "PERSONAL-PositionDepartment":        "PERSONAL-POSITION",
    "PERSONAL-Address":                   "PERSONAL-ADDRESS",
    "PERSONAL-Date":                      "PERSONAL-DATE",
    "PERSONAL-Location":                  "PERSONAL-LOCATION",
    "PERSONAL-PersonalInformation":       "PERSONAL-INFO",
    "PERSONAL-Company":                   "PERSONAL-COMPANY",
    "PERSONAL-ArtisticActivity":          "PERSONAL-ARTISTIC",
    "PERSONAL-Degree":                    "PERSONAL-DEGREE",
    "PERSONAL-Time":                      "PERSONAL-TIME",
    "PERSONAL-LicensePlate":              "PERSONAL-LICENSE",
    "PERSONAL-Job":                       "PERSONAL-JOB",
    "PERSONAL-Vehicle":                   "PERSONAL-VEHICLE",
    "PERSONAL-Faculty":                   "PERSONAL-FACULTY",
    "PERSONAL-Family":                    "PERSONAL-FAMILY",
    "PERSONAL-Other":                     "PERSONAL-OTHER",
}

DIST_BUCKETS      = [1, 2, 3, 4, 5, 8, 16, 32, 64]
MIN_CLUSTER_SCORE = 0.3
NER_UNION_MARGIN  = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# I/O — load gold documents
# ─────────────────────────────────────────────────────────────────────────────

def load_test_files(json_dir: Path) -> List[str]:
    # Load test file list from split_info.json in the dataset directory
    split_path = json_dir / "split_info.json"
    with open(split_path, encoding="utf-8") as f:
        splits = json.load(f)
    return splits["test"]


def find_json(name: str, json_dir: Path) -> Optional[Path]:
    p = json_dir / f"{name}.json"
    if p.exists():
        return p
    for p2 in json_dir.glob(f"*{name}.json"):
        return p2
    return None


def load_document(path: Path) -> Optional[Dict]:
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"  ERROR: {path.name}: {e}")
        return None

    tokens       = raw.get("tokens", [])
    ner_raw      = raw.get("ner", [])
    clusters_raw = raw.get("clusters", [])

    if not tokens:
        return None

    ner_map: Dict[Tuple[int, int], str] = {}
    for entry in ner_raw:
        if len(entry) == 3 and str(entry[2]) in VALID_NER_CLASSES:
            ner_map[(int(entry[0]), int(entry[1]))] = str(entry[2])

    gold_clusters = []
    for cluster in clusters_raw:
        counts: Dict[str, int] = defaultdict(int)
        for sp in cluster:
            key = (int(sp[0]), int(sp[1]))
            if key in ner_map:
                counts[ner_map[key]] += 1
        if not counts:
            continue
        dominant = max(counts, key=counts.__getitem__)
        filtered = [
            (int(sp[0]), int(sp[1])) for sp in cluster
            if ner_map.get((int(sp[0]), int(sp[1]))) == dominant
        ]
        if len(filtered) >= 2:
            gold_clusters.append(filtered)

    text        = " ".join(tokens)
    char_starts = []
    pos         = 0
    for tok in tokens:
        char_starts.append(pos)
        pos += len(tok) + 1

    return {
        "doc_key":       raw.get("doc_key", path.stem),
        "tokens":        tokens,
        "text":          text,
        "char_starts":   char_starts,
        "ner_map":       ner_map,
        "gold_clusters": gold_clusters,
    }


def load_documents(json_dir: Path) -> List[Dict]:
    test_files = load_test_files(json_dir)
    docs       = []
    for name in test_files:
        path = find_json(name, json_dir)
        if path is None:
            print(f"  WARNING: {name} not found")
            continue
        doc = load_document(path)
        if doc:
            docs.append(doc)
    return docs

# Gold standard — entities with cluster_id per document
def build_gold_entities(doc: Dict) -> List[Dict]:
    # Build gold entity list with consistent cluster IDs
    # Entities outside clusters receive singleton IDs based on their exact text
    span_to_cluster: Dict[Tuple[int, int], int] = {}
    cluster_counter = defaultdict(int)

    for cluster in doc["gold_clusters"]:
        classe = doc["ner_map"].get(cluster[0])
        if classe is None:
            continue
        cluster_counter[classe] += 1
        cid = cluster_counter[classe]
        for sp in cluster:
            span_to_cluster[sp] = cid

    gold_entities            = []
    singleton_text_to_id: Dict[Tuple[str, str], int] = {}

    for (s, e), classe in sorted(doc["ner_map"].items()):
        texto = " ".join(doc["tokens"][s:e + 1]).lower()
        if (s, e) in span_to_cluster:
            cid = span_to_cluster[(s, e)]
        else:
            key = (classe, texto)
            if key not in singleton_text_to_id:
                cluster_counter[classe] += 1
                singleton_text_to_id[key] = cluster_counter[classe]
            cid = singleton_text_to_id[key]

        gold_entities.append({
            "start": s, "end": e, "classe": classe,
            "texto": " ".join(doc["tokens"][s:e + 1]),
            "cluster_id": cid,
        })

    return gold_entities

# Coreference model architecture
class CorefModel(nn.Module):
    def __init__(self, model_dir, ffnn_dim, max_span_width,
                 max_antecedents, max_cluster_size):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(str(model_dir))
        H = self.encoder.config.hidden_size
        self.hidden           = H
        self.max_span_width   = max_span_width
        self.max_antecedents  = max_antecedents
        self.max_cluster_size = max_cluster_size

        self.width_emb = nn.Embedding(max_span_width + 1, 20)
        self.head_attn = nn.Linear(H, 1)
        span_dim       = H * 3 + 20

        self.pos_emb        = nn.Embedding(20, 32)
        self.mention_scorer = nn.Sequential(
            nn.Linear(span_dim + 32, ffnn_dim), nn.ReLU(), nn.Dropout(0.0),
            nn.Linear(ffnn_dim, ffnn_dim),       nn.ReLU(), nn.Dropout(0.0),
            nn.Linear(ffnn_dim, 1),
        )

        n_dist        = len(DIST_BUCKETS) + 1
        self.dist_emb = nn.Embedding(n_dist, 20)
        self.register_buffer(
            "dist_bucket_boundaries",
            torch.tensor(DIST_BUCKETS, dtype=torch.long)
        )
        self.ant_scorer = nn.Sequential(
            nn.Linear(span_dim * 3 + 20, ffnn_dim), nn.ReLU(), nn.Dropout(0.0),
            nn.Linear(ffnn_dim, ffnn_dim),            nn.ReLU(), nn.Dropout(0.0),
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

# Model loading
def load_ner_model(model_dir: Path, device):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), add_prefix_space=True)
    model     = AutoModelForTokenClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()
    return tokenizer, model


def load_coref_model(model_dir: Path, device):
    model_dir = Path(model_dir)
    with open(model_dir / "coref_config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    model = CorefModel(
        model_dir=str(model_dir),
        ffnn_dim=cfg["ffnn_dim"],
        max_span_width=cfg["max_span_width"],
        max_antecedents=cfg["max_antecedents"],
        max_cluster_size=cfg["max_cluster_size"],
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

    return model, tokenizer, max_tokens, stride

# Sliding windows & embeddings
def sliding_windows(tokens, tokenizer, max_tokens, stride):
    flat = []
    for idx, tok in enumerate(tokens):
        subs = tokenizer.encode(tok, add_special_tokens=False) or [tokenizer.unk_token_id]
        flat.extend((idx, s) for s in subs)

    cls_id      = tokenizer.cls_token_id or tokenizer.bos_token_id
    sep_id      = tokenizer.sep_token_id or tokenizer.eos_token_id
    pad_id      = tokenizer.pad_token_id
    max_content = max_tokens - 2

    windows, start = [], 0
    while start < len(flat):
        end   = min(start + max_content, len(flat))
        chunk = flat[start:end]
        ids   = [cls_id] + [s for _, s in chunk] + [sep_id]
        offs  = [-1]     + [o for o, _ in chunk]  + [-1]
        mask  = [1] * len(ids)
        pad   = max_tokens - len(ids)
        ids  += [pad_id] * pad
        mask += [0]      * pad
        offs += [-1]     * pad
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

# NER inference
@torch.no_grad()
def predict_ner_spans(text, tokenizer, model, device):
    # Run NER and return char-level spans with mapped gold class labels
    id2label       = model.config.id2label
    SPACE_PREFIXES = [" ", "▁", "Ġ"]

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512, stride=64,
        return_overflowing_tokens=True, return_offsets_mapping=True, padding=True
    ).to(device)
    offset_mapping = inputs.pop("offset_mapping")
    inputs.pop("overflow_to_sample_mapping", None)

    entidades_brutas = []
    for i in range(len(inputs["input_ids"])):
        outputs     = model(input_ids=inputs["input_ids"][i:i + 1],
                            attention_mask=inputs["attention_mask"][i:i + 1])
        predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
        tokens_bert = tokenizer.convert_ids_to_tokens(inputs["input_ids"][i])
        offsets     = offset_mapping[i]

        temp_entity = {"start": None, "end": None, "label": None}
        for idx, (token, pred_id, offset) in enumerate(zip(tokens_bert, predictions, offsets)):
            label_name = id2label[pred_id]
            start_char, end_char = int(offset[0]), int(offset[1])
            if start_char == end_char:
                continue
            tag_base    = label_name.split('-', 1)[1] if '-' in label_name else None
            comeca_nova = any(token.startswith(p) for p in SPACE_PREFIXES)

            if label_name.startswith("B-"):
                if temp_entity["label"]:
                    entidades_brutas.append(temp_entity)
                temp_entity = {"start": start_char, "end": end_char, "label": tag_base}
            elif label_name.startswith("I-") and temp_entity["label"] == tag_base:
                temp_entity["end"] = end_char
            elif not comeca_nova and temp_entity["label"] is not None:
                temp_entity["end"] = end_char
            else:
                if temp_entity["label"]:
                    entidades_brutas.append(temp_entity)
                temp_entity = {"start": None, "end": None, "label": None}

        if temp_entity["label"]:
            entidades_brutas.append(temp_entity)

    entidades_brutas.sort(key=lambda x: (x["start"], -x["end"]))
    entidades_finais = []
    for atual in entidades_brutas:
        adicionar = True
        for i, sel in enumerate(entidades_finais):
            inter = max(0, min(atual["end"], sel["end"]) - max(atual["start"], sel["start"]))
            if inter > 0 and atual["label"] == sel["label"]:
                if (atual["end"] - atual["start"]) > (sel["end"] - sel["start"]):
                    entidades_finais[i] = atual
                adicionar = False
                break
        if adicionar:
            entidades_finais.append(atual)

    result = []
    for e in entidades_finais:
        if not e["label"]:
            continue
        classe_gold = NER_LABEL_TO_GOLD_CLASS.get(e["label"])
        if classe_gold is None or classe_gold not in VALID_NER_CLASSES:
            continue
        result.append((e["start"], e["end"], classe_gold))
    return result


def char_spans_to_token_spans(char_spans, char_starts, tokens):
    # Convert char-level spans to token-level spans
    result = []
    for (sc, ec, classe) in char_spans:
        s_tok = e_tok = None
        for ti, cs in enumerate(char_starts):
            ce = cs + len(tokens[ti])
            if s_tok is None and ce > sc:
                s_tok = ti
            if cs < ec:
                e_tok = ti
        if s_tok is not None and e_tok is not None:
            result.append((s_tok, e_tok, classe))
    return result


# Coreference inference
@torch.no_grad()
def predict_clusters_from_spans(model, tokens, ner_spans, tokenizer,
                                 device, max_tokens, stride):
    all_spans = sorted({(s, e) for s, e, _ in ner_spans})
    if not all_spans:
        return [], {}

    span_classe = {(s, e): c for s, e, c in ner_spans}
    windows     = sliding_windows(tokens, tokenizer, max_tokens, stride)
    if not windows:
        return [], span_classe

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

    threshold     = max(model.epsilon.item(), MIN_CLUSTER_SCORE)
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
            if span_classe.get(si) == span_classe.get(sj):
                union(i, best_j)

    grupos: Dict = defaultdict(list)
    for i in range(len(all_spans)):
        grupos[find(i)].append(all_spans[i])
    clusters = [v for v in grupos.values() if len(v) >= 2]
    return clusters, span_classe


# Predicted entities with cluster IDs
def build_pred_entities(pred_spans, clusters, span_classe, tokens):
    span_to_cluster = {}
    cluster_counter = defaultdict(int)

    for cluster in clusters:
        classe = span_classe.get(cluster[0])
        if classe is None:
            continue
        cluster_counter[classe] += 1
        cid = cluster_counter[classe]
        for sp in cluster:
            span_to_cluster[sp] = cid

    pred_entities = []
    for (s, e, classe) in pred_spans:
        if (s, e) in span_to_cluster:
            cid = span_to_cluster[(s, e)]
        else:
            cluster_counter[classe] += 1
            cid = cluster_counter[classe]
        pred_entities.append({
            "start": s, "end": e, "classe": classe, "cluster_id": cid,
            "texto": " ".join(tokens[s:e + 1]),
        })
    return pred_entities


# Evaluation — span-by-span comparison
def evaluate_document(gold_entities, pred_entities):
    gold_by_span = {(e["start"], e["end"]): e for e in gold_entities}
    pred_by_span = {(e["start"], e["end"]): e for e in pred_entities}

    correctas, perdidas, incorrectas = [], [], []
    matched_pred_spans = set()

    gold_cluster_to_pred_ids = defaultdict(set)
    pred_cluster_to_gold_ids = defaultdict(set)

    for span, gent in gold_by_span.items():
        pent = pred_by_span.get(span)
        if pent is None:
            perdidas.append(gent)
            continue
        matched_pred_spans.add(span)
        gold_key = (gent["classe"], gent["cluster_id"])
        pred_key = (pent["classe"], pent["cluster_id"])
        gold_cluster_to_pred_ids[gold_key].add(pred_key)
        pred_cluster_to_gold_ids[pred_key].add(gold_key)

        if pent["classe"] != gent["classe"]:
            incorrectas.append({"gold": gent, "pred": pent, "motivo": "classe_errada"})
        else:
            correctas.append({"gold": gent, "pred": pent})

    consistencia_ids_corretos = []
    for entry in correctas:
        gold_key = (entry["gold"]["classe"], entry["gold"]["cluster_id"])
        pred_key = (entry["pred"]["classe"], entry["pred"]["cluster_id"])
        if (len(gold_cluster_to_pred_ids[gold_key]) == 1 and
                len(pred_cluster_to_gold_ids[pred_key]) == 1):
            consistencia_ids_corretos.append(entry)
        else:
            incorrectas.append({**entry, "motivo": "id_inconsistente"})

    falsos_positivos = [pent for span, pent in pred_by_span.items()
                        if span not in matched_pred_spans]

    return {
        "correctas":        consistencia_ids_corretos,
        "perdidas":         perdidas,
        "incorrectas":      incorrectas,
        "falsos_positivos": falsos_positivos,
    }


# Main
def main():
    parser = argparse.ArgumentParser(
        description="End-to-end evaluation of NER + Coreference pseudonymization pipeline"
    )
    parser.add_argument("--ner_model",   type=str,
                        default="models/general_model/ner/xlm_roberta",
                        help="Path to the trained NER model")
    parser.add_argument("--coref_model", type=str,
                        default="models/general_model/coref/spanbert_pt",
                        help="Path to the trained coreference model")
    parser.add_argument("--json_dir",    type=str,
                        default="data/coreference_dataset",
                        help="Directory with JSON document files and split_info.json")
    parser.add_argument("--output",      type=str,
                        default="results/e2e/metrics.json",
                        help="Path to save the evaluation results JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load models
    print(f"Loading NER model ({args.ner_model})...")
    tok_ner, model_ner = load_ner_model(Path(args.ner_model), device)

    print(f"Loading coreference model ({args.coref_model})...")
    model_coref, tok_coref, max_tokens, stride = load_coref_model(
        Path(args.coref_model), device
    )

    # Load test documents
    print(f"\nLoading test documents ({args.json_dir})...")
    docs = load_documents(Path(args.json_dir))
    print(f"  {len(docs)} documents loaded\n")

    total         = defaultdict(int)
    detalhes_docs = []

    for doc in tqdm(docs, desc="Evaluating E2E pipeline", unit="doc"):
        gold_entities = build_gold_entities(doc)

        char_spans = predict_ner_spans(doc["text"], tok_ner, model_ner, device)
        ner_spans  = char_spans_to_token_spans(
            char_spans, doc["char_starts"], doc["tokens"]
        )

        clusters, span_classe = predict_clusters_from_spans(
            model_coref, doc["tokens"], ner_spans,
            tok_coref, device, max_tokens, stride
        )
        pred_entities = build_pred_entities(
            ner_spans, clusters, span_classe, doc["tokens"]
        )

        resultado = evaluate_document(gold_entities, pred_entities)

        total["correctas"]        += len(resultado["correctas"])
        total["perdidas"]         += len(resultado["perdidas"])
        total["incorrectas"]      += len(resultado["incorrectas"])
        total["falsos_positivos"] += len(resultado["falsos_positivos"])
        total["gold_total"]       += len(gold_entities)
        total["pred_total"]       += len(pred_entities)

        detalhes_docs.append({
            "doc_key":     doc["doc_key"],
            "n_gold":      len(gold_entities),
            "n_pred":      len(pred_entities),
            "n_correctas": len(resultado["correctas"]),
            "n_perdidas":  len(resultado["perdidas"]),
            "n_incorrectas": len(resultado["incorrectas"]),
            "n_falsos_pos":  len(resultado["falsos_positivos"]),
            "exemplos_perdidas": [
                {"texto": e["texto"], "classe": e["classe"]}
                for e in resultado["perdidas"][:3]
            ],
            "exemplos_incorrectas": [
                {
                    "texto_gold":  e["gold"]["texto"],
                    "texto_pred":  e["pred"]["texto"],
                    "classe_gold": e["gold"]["classe"],
                    "classe_pred": e["pred"]["classe"],
                    "motivo":      e["motivo"],
                }
                for e in resultado["incorrectas"][:3]
            ],
            "exemplos_falsos_pos": [
                {"texto": fp["texto"], "classe": fp["classe"]}
                for fp in resultado["falsos_positivos"][:3]
            ],
        })

    # Compute final metrics
    tp = total["correctas"]
    fn = total["perdidas"]         + total["incorrectas"]
    fp = total["incorrectas"]      + total["falsos_positivos"]

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)

    print()
    print("=" * 70)
    print("  END-TO-END EVALUATION — PSEUDONYMIZATION (NER + COREFERENCE)")
    print("=" * 70)
    print(f"  {'Metric':<45} {'Value':>15}")
    print(f"  {'-'*62}")
    print(f"  {'Correctly pseudonymized entities':<45} {tp:>15}")
    print(f"  {'Lost entities (not detected)':<45} {total['perdidas']:>15}")
    print(f"  {'Incorrect substitutions':<45} {total['incorrectas']:>15}")
    print(f"  {'False positives (no gold match)':<45} {total['falsos_positivos']:>15}")
    print(f"  {'-'*62}")
    print(f"  {'Precision (%)':<45} {precision*100:>14.2f}%")
    print(f"  {'Recall (%)':<45} {recall*100:>14.2f}%")
    print(f"  {'F1-Score (%)':<45} {f1*100:>14.2f}%")
    print("=" * 70)

    print()
    print("=" * 70)
    print("  ERROR EXAMPLES (first 3 per document)")
    print("=" * 70)
    for d in detalhes_docs:
        if not any([d["exemplos_perdidas"], d["exemplos_incorrectas"],
                    d["exemplos_falsos_pos"]]):
            continue
        print(f"\n  DOC: {d['doc_key']}")
        if d["exemplos_perdidas"]:
            print("    Lost (gold not detected):")
            for ex in d["exemplos_perdidas"]:
                print(f"      [{ex['classe']}] \"{ex['texto']}\"")
        if d["exemplos_incorrectas"]:
            print("    Incorrect (wrong class or inconsistent ID):")
            for ex in d["exemplos_incorrectas"]:
                print(f"      [{ex['motivo']}] gold=\"{ex['texto_gold']}\" "
                      f"({ex['classe_gold']}) -> pred=\"{ex['texto_pred']}\" "
                      f"({ex['classe_pred']})")
        if d["exemplos_falsos_pos"]:
            print("    False positives (predicted without gold match):")
            for ex in d["exemplos_falsos_pos"]:
                print(f"      [{ex['classe']}] \"{ex['texto']}\"")
    print("=" * 70)

    # Save results
    output_data = {
        "ner_model":   str(args.ner_model),
        "coref_model": str(args.coref_model),
        "n_docs":      len(docs),
        "totals": {
            "gold_entities":                 total["gold_total"],
            "predicted_entities":            total["pred_total"],
            "correctly_pseudonymized":       tp,
            "lost":                          total["perdidas"],
            "incorrect_substitutions":       total["incorrectas"],
            "false_positives":               total["falsos_positivos"],
        },
        "metrics": {
            "precision": round(precision * 100, 2),
            "recall":    round(recall * 100, 2),
            "f1_score":  round(f1 * 100, 2),
        },
        "per_document": detalhes_docs,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved in: {output_path}")


if __name__ == "__main__":
    main()