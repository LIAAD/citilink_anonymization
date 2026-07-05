import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm", "-q"])
    from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

VALID_NER_CLASSES = {
    "PERSONAL-NAME", "PERSONAL-ADMIN", "PERSONAL-POSITION", "PERSONAL-LOCATION",
    "PERSONAL-ADDRESS", "PERSONAL-DATE", "PERSONAL-OTHER", "PERSONAL-COMPANY",
    "PERSONAL-INFO", "PERSONAL-ARTISTIC", "PERSONAL-JOB", "PERSONAL-DEGREE",
    "PERSONAL-TIME", "PERSONAL-FACULTY", "PERSONAL-FAMILY", "PERSONAL-LICENSE",
    "PERSONAL-VEHICLE",
}

DIST_BUCKETS     = [1, 2, 3, 4, 5, 8, 16, 32, 64]
MIN_CLUSTER_SCORE = 0.3
NER_UNION_MARGIN  = 0.5

TEST_FILES = [
    'Campomaior_cm_014_2023-07-05', 'Campomaior_cm_018_2024-10-02',
    'Campomaior_cm_011_2023-06-07', 'Campomaior_cm_024_2022-12-07',
    'Campomaior_cm_001_2023-01-04', 'Fundao_cm_005_2022-22-04',
    'Campomaior_cm_025_2022-12-21', 'Covilha_cm_006_2023-04-21',
    'Porto_cm_005_2022-05-30',      'Covilha_cm_021_2024-12-12',
    'Guimaraes_cm_020_2021-12-06',  'Alandroal_cm_006_2022-03-16',
    'Fundao_cm_006_2024-08-04',     'Guimaraes_cm_018_2024-10-28',
    'Alandroal_cm_015_2024-06-19',  'Covilha_cm_005_2024-03-22',
    'Porto_cm_004_2021-12-06',      'Guimaraes_cm_010_2022-05-19',
    'Porto_cm_017_2022-06-27',      'Porto_cm_062_2024-07-08',
    'Fundao_cm_003_2024-16-02',     'Fundao_cm_012_2023-25-09',
    'Guimaraes_cm_009_2024-05-06',  'Guimaraes_cm_016_2024-09-30',
]

# Model Architecture
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

# I/O
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
        print(f"  Error reading {path.name}: {e}")
        return None

    tokens       = raw.get("tokens", [])
    ner_raw      = raw.get("ner", [])
    clusters_raw = raw.get("clusters", [])

    if not tokens or not clusters_raw:
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

    if not gold_clusters:
        return None

    return {
        "doc_key":       raw.get("doc_key", path.stem),
        "tokens":        tokens,
        "ner_map":       ner_map,
        "gold_clusters": gold_clusters,
    }


def load_test_documents(json_dir: Path) -> List[Dict]:
    docs = []
    for name in TEST_FILES:
        path = find_json(name, json_dir)
        if path is None:
            print(f"  WARNING: {name} not found")
            continue
        doc = load_document(path)
        if doc:
            docs.append(doc)
    return docs


# Model Loading
def load_model(model_dir: Path, device):
    model_dir = Path(model_dir)
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

    return model, tokenizer, max_tokens, stride, cfg

# Sliding Windows
def sliding_windows(tokens, tokenizer, max_tokens, stride):
    flat = []
    for idx, tok in enumerate(tokens):
        subs = tokenizer.encode(tok, add_special_tokens=False) or [tokenizer.unk_token_id]
        flat.extend((idx, s) for s in subs)

    cls_id      = tokenizer.cls_token_id
    sep_id      = tokenizer.sep_token_id
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

# Inference
@torch.no_grad()
def predict_clusters(model, doc, tokenizer, device, max_tokens, stride, cfg):
    ner_map       = doc["ner_map"]
    tokens        = doc["tokens"]
    gold_clusters = doc["gold_clusters"]
    windows       = sliding_windows(tokens, tokenizer, max_tokens, stride)

    if not windows or not ner_map or not gold_clusters:
        return []

    gold_span_set = {sp for cl in gold_clusters for sp in cl}
    all_spans     = sorted(gold_span_set)
    if not all_spans:
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
        if ra == rb:
            return
        if cl_size[ra] + cl_size[rb] > model.max_cluster_size:
            return
        parent[ra]   = rb
        cl_size[rb] += cl_size[ra]

    epsilon       = cfg.get("epsilon", model.epsilon.item())
    threshold     = max(epsilon, MIN_CLUSTER_SCORE)
    ner_threshold = threshold + NER_UNION_MARGIN

    for i in range(1, len(all_spans)):
        j_max = min(i, model.max_antecedents)
        js    = list(range(i - j_max, i))
        if not js:
            continue
        js_t   = torch.tensor(js, dtype=torch.long, device=device)
        pair   = model._pair_vec(sv, i, js_t)
        scores = model.ant_scorer(pair).squeeze(-1) + ms[i] + ms[js_t]
        best_l = scores.argmax().item()
        best_s = scores[best_l].item()
        best_j = js[best_l]
        if best_s > ner_threshold:
            si, sj = all_spans[i], all_spans[best_j]
            if ner_map.get(si) == ner_map.get(sj):
                union(i, best_j)

    clusters_out: Dict[int, list] = defaultdict(list)
    for i in range(len(all_spans)):
        clusters_out[find(i)].append(all_spans[i])
    return [v for v in clusters_out.values() if len(v) >= 2]

# Metrics
def _sets(clusters):
    return [set(map(tuple, c)) for c in clusters]


def muc_score(pred, gold):
    def _part(cl, others):
        rem, parts = set(cl), []
        for o in others:
            inter = rem & o
            if inter: parts.append(inter); rem -= inter
        if rem: parts.append(rem)
        return parts

    def _muc(key, resp):
        rn = rd = pn = pd = 0
        for k in key:
            rd += len(k) - 1; rn += len(k) - len(_part(k, resp))
        for r in resp:
            pd += len(r) - 1; pn += len(r) - len(_part(r, key))
        return rn, rd, pn, pd

    key, resp   = _sets(gold), _sets(pred)
    rn, rd, pn, pd = _muc(key, resp)
    r = rn / rd if rd else 0.0
    p = pn / pd if pd else 0.0
    return {"precision": p, "recall": r, "f1": 2*p*r/(p+r) if p+r else 0.0}


def b3_score(pred, gold):
    key, resp = _sets(gold), _sets(pred)
    km, rm = {}, {}
    for i, c in enumerate(key):
        for m in c: km[m] = i
    for i, c in enumerate(resp):
        for m in c: rm[m] = i
    all_m = set(km) | set(rm)
    rn = rd = pn = pd = 0
    for m in all_m:
        if m in km:
            gc    = key[km[m]]; rc = resp[rm[m]] if m in rm else {m}
            inter = gc & rc
            rn += len(inter) / len(gc); rd += 1
            pn += len(inter) / len(rc); pd += 1
    r = rn / rd if rd else 0.0
    p = pn / pd if pd else 0.0
    return {"precision": p, "recall": r, "f1": 2*p*r/(p+r) if p+r else 0.0}


def ceaf_e_score(pred, gold):
    from scipy.optimize import linear_sum_assignment
    key, resp = _sets(gold), _sets(pred)
    if not key or not resp:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    sim     = np.array([[len(k & r) / len(k | r) if k | r else 0.0
                         for r in resp] for k in key])
    ri, ci  = linear_sum_assignment(-sim)
    s       = sum(sim[r, c] for r, c in zip(ri, ci))
    p = s / len(resp); r = s / len(key)
    return {"precision": p, "recall": r, "f1": 2*p*r/(p+r) if p+r else 0.0}


def lea_score(pred, gold):
    def lnk(c): n = len(c); return n * (n - 1) / 2

    def _p(key, resp):
        n = d = 0
        for r in resp:
            rs = set(map(tuple, r)); imp = lnk(rs)
            if not imp: continue
            res = sum(lnk(rs & set(map(tuple, k))) for k in key)
            n += imp * (res / imp); d += imp
        return n / d if d else 0.0

    def _r(key, resp):
        n = d = 0
        for k in key:
            ks = set(map(tuple, k)); imp = lnk(ks)
            if not imp: continue
            res = sum(lnk(ks & set(map(tuple, r))) for r in resp)
            n += imp * (res / imp); d += imp
        return n / d if d else 0.0

    p = _p(gold, pred); r = _r(gold, pred)
    return {"precision": p, "recall": r, "f1": 2*p*r/(p+r) if p+r else 0.0}


def conll_f1(muc, b3, ceafe):
    return (muc["f1"] + b3["f1"] + ceafe["f1"]) / 3


def prefix_clusters(clusters, doc_key):
    return [[(doc_key, s, e) for s, e in cl] for cl in clusters]


def evaluate_by_class(model, docs, tokenizer, device, max_tokens, stride, cfg):
    gold_por_classe: Dict[str, list] = defaultdict(list)
    pred_por_classe: Dict[str, list] = defaultdict(list)

    for doc in docs:
        dk      = doc["doc_key"]
        ner_map = doc["ner_map"]
        for cluster in doc["gold_clusters"]:
            if not cluster:
                continue
            classe = ner_map.get(cluster[0], "UNKNOWN")
            gold_por_classe[classe].append([(dk, s, e) for s, e in cluster])
        pred_clusters = predict_clusters(
            model, doc, tokenizer, device, max_tokens, stride, cfg
        )
        for cluster in pred_clusters:
            if not cluster:
                continue
            classe = ner_map.get(cluster[0], "UNKNOWN")
            pred_por_classe[classe].append([(dk, s, e) for s, e in cluster])

    resultados = {}
    for classe in sorted(set(gold_por_classe) | set(pred_por_classe)):
        gold_cl = gold_por_classe.get(classe, [])
        pred_cl = pred_por_classe.get(classe, [])
        if not gold_cl:
            continue
        muc   = muc_score(pred_cl, gold_cl)
        b3    = b3_score(pred_cl, gold_cl)
        ceafe = ceaf_e_score(pred_cl, gold_cl)
        lea   = lea_score(pred_cl, gold_cl)
        resultados[classe] = {
            "n_clusters_gold": len(gold_cl),
            "n_clusters_pred": len(pred_cl),
            "MUC":      muc,
            "B3":       b3,
            "CEAF_e":   ceafe,
            "CoNLL_F1": conll_f1(muc, b3, ceafe),
            "LEA":      lea,
        }
    return resultados


# Main
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate coreference model on test set"
    )
    parser.add_argument("--config",     type=str, required=True,
                        help="Config name (e.g., spanbert_pt, bertimbau_coref)")
    parser.add_argument("--model_path", type=str,
                        help="Path to trained model (overrides default)")
    parser.add_argument("--json_dir",   type=str, default="data/coreference_dataset",
                        help="Directory with JSON document files")
    args = parser.parse_args()

    # Load configuration
    config_path = "config/training_configs.json"
    with open(config_path, "r") as f:
        all_configs = json.load(f)

    if args.config not in all_configs:
        print(f"Error: Config '{args.config}' not found in {config_path}")
        return

    params = all_configs[args.config]

    # Define paths
    model_dir   = Path(args.model_path) if args.model_path \
                  else Path(f"models/general_model/coref/{args.config}")
    results_dir = Path(f"results/coreference/general_model/{args.config}")
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Evaluating {args.config} from: {model_dir}")

    # Load model
    model, tokenizer, max_tokens, stride, cfg = load_model(model_dir, device)

    # Load test documents
    json_dir = Path(args.json_dir)
    docs     = load_test_documents(json_dir)
    print(f"Test documents loaded: {len(docs)}")

    # Run inference and collect clusters
    flat_gold, flat_pred = [], []
    for doc in tqdm(docs, desc="Evaluating", unit="doc"):
        dk = doc["doc_key"]
        flat_gold += prefix_clusters(doc["gold_clusters"], dk)
        flat_pred += prefix_clusters(
            predict_clusters(model, doc, tokenizer, device, max_tokens, stride, cfg), dk
        )

    # Compute overall metrics
    muc   = muc_score(flat_pred, flat_gold)
    b3    = b3_score(flat_pred, flat_gold)
    ceafe = ceaf_e_score(flat_pred, flat_gold)
    lea   = lea_score(flat_pred, flat_gold)
    cf1   = conll_f1(muc, b3, ceafe)

    print(f"\nResults for {args.config}:")
    print(f"  {'Metric':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*44}")
    for name, v in [("MUC", muc), ("B3", b3), ("CEAF-e", ceafe), ("LEA", lea)]:
        print(f"  {name:<12} {v['precision']:>10.4f} {v['recall']:>10.4f} {v['f1']:>10.4f}")
    print(f"  {'CoNLL F1':<12} {'':>10} {'':>10} {cf1:>10.4f}")

    # Compute per-class metrics
    print("\nComputing per-class metrics...")
    por_classe = evaluate_by_class(
        model, docs, tokenizer, device, max_tokens, stride, cfg
    )

    print(f"\n  {'Class':<30} {'Gold':>6} {'Pred':>6} {'CoNLL F1':>10}")
    print(f"  {'-'*56}")
    for classe, m in sorted(por_classe.items(), key=lambda x: -x[1]["CoNLL_F1"]):
        print(f"  {classe:<30} {m['n_clusters_gold']:>6} "
              f"{m['n_clusters_pred']:>6} {m['CoNLL_F1']:>10.4f}")

    # Save results to JSON
    results = {
        "config":      args.config,
        "model_path":  str(model_dir),
        "n_docs":      len(docs),
        "metrics": {
            "MUC":      muc,
            "B3":       b3,
            "CEAF_e":   ceafe,
            "CoNLL_F1": cf1,
            "LEA":      lea,
        },
        "per_class": por_classe,
    }
    out_path = results_dir / f"metrics_{args.config}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation complete. Results saved in: {out_path}")


if __name__ == "__main__":
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    main()