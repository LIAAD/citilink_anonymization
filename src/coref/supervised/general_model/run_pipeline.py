import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

try:
    from tqdm import tqdm
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm", "-q"])
    from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

VALID_NER_CLASSES = {
    "PERSONAL-NAME", "PERSONAL-ADMIN", "PERSONAL-POSITION", "PERSONAL-LOCATION",
    "PERSONAL-ADDRESS", "PERSONAL-DATE", "PERSONAL-OTHER", "PERSONAL-COMPANY",
    "PERSONAL-INFO", "PERSONAL-ARTISTIC", "PERSONAL-JOB", "PERSONAL-DEGREE",
    "PERSONAL-TIME", "PERSONAL-FACULTY", "PERSONAL-FAMILY", "PERSONAL-LICENSE",
    "PERSONAL-VEHICLE",
}

DIST_BUCKETS      = [1, 2, 3, 4, 5, 8, 16, 32, 64]
MIN_CLUSTER_SCORE = 0.3
NER_UNION_MARGIN  = 0.5


def load_splits(data_dir: Path):
    # Load train/val/test splits from split_info.json
    split_path = Path(data_dir) / "split_info.json"
    with open(split_path, encoding="utf-8") as f:
        splits = json.load(f)
    return splits["train"], splits["val"], splits["test"]


def fmt(s):
    return time.strftime("%H:%M:%S", time.gmtime(s))


# Model Architecture
class CorefModel(nn.Module):
    def __init__(self, model_name, ffnn_dim, max_span_width,
                 max_antecedents, max_cluster_size, dropout=0.3):
        super().__init__()
        self.encoder          = AutoModel.from_pretrained(str(model_name))
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

    def mention_ranking_loss(self, span_vecs, gold_spans_list,
                              gold_clusters, device):
        span2c = {}
        for c_i, cl in enumerate(gold_clusters):
            for sp in cl:
                span2c[tuple(sp)] = c_i

        n = len(gold_spans_list)
        if n < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        gs_t     = torch.tensor([s for s, e in gold_spans_list],
                                dtype=torch.long, device=device)
        doc_len  = gs_t.max().item() + 1
        m_scores = self.mention_score_with_pos(span_vecs, gs_t, doc_len)

        total = torch.tensor(0.0, device=device)
        n_all = 0

        for i in range(1, n):
            si    = gold_spans_list[i]
            ci    = span2c.get(si, -1)
            j_max = min(i, self.max_antecedents)
            js    = list(range(i - j_max, i))
            if not js:
                continue
            js_t  = torch.tensor(js, dtype=torch.long, device=device)
            pair  = self._pair_vec(span_vecs, i, js_t)
            a_sc  = (self.ant_scorer(pair).squeeze(-1)
                     + m_scores[i] + m_scores[js_t])
            eps      = self.epsilon.squeeze().unsqueeze(0)
            all_sc   = torch.cat([eps, a_sc], dim=0)
            pos_mask = torch.tensor(
                [span2c.get(gold_spans_list[j], -2) == ci and ci != -1
                 for j in js], dtype=torch.bool, device=device)
            log_norm = torch.logsumexp(all_sc, dim=0)
            if pos_mask.any():
                pos_idx   = pos_mask.nonzero(as_tuple=True)[0] + 1
                log_numer = torch.logsumexp(all_sc[pos_idx], dim=0)
                total    += log_norm - log_numer
            else:
                total += log_norm - all_sc[0]
            n_all += 1

        return total / max(n_all, 1)

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


def load_split(file_list, json_dir: Path, split_name: str) -> List[Dict]:
    docs    = []
    missing = 0
    for name in file_list:
        path = find_json(name, json_dir)
        if path is None:
            print(f"  WARNING [{split_name}]: {name} not found")
            missing += 1
            continue
        doc = load_document(path)
        if doc:
            docs.append(doc)
    print(f"  [{split_name}] {len(docs)} docs loaded | {missing} missing")
    return docs

# Sliding Windows & Embeddings
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

# Training
def train_one_doc(model, doc, tokenizer, optimizer, scheduler,
                  device, max_tokens, stride, grad_acc):
    windows = sliding_windows(doc["tokens"], tokenizer, max_tokens, stride)
    if not windows:
        return 0.0

    gold_clusters   = doc["gold_clusters"]
    gold_span_set   = {sp for cl in gold_clusters for sp in cl}
    gold_spans_list = sorted(gold_span_set)
    if not gold_spans_list:
        return 0.0

    max_tok  = len(doc["tokens"])
    H        = model.hidden
    n_blocks = max(1, (len(windows) + grad_acc - 1) // grad_acc)

    with torch.no_grad():
        cached_hidden = []
        for w in windows:
            inp  = torch.tensor([w["input_ids"]],      dtype=torch.long)
            mask = torch.tensor([w["attention_mask"]], dtype=torch.long)
            cached_hidden.append(model.encode_window(inp, mask, device)[0].detach())

    total_loss_val = 0.0
    for b_start in range(0, len(windows), grad_acc):
        b_end    = b_start + grad_acc
        b_hidden, b_offs = [], []
        for w in windows[b_start:b_end]:
            inp  = torch.tensor([w["input_ids"]],      dtype=torch.long)
            mask = torch.tensor([w["attention_mask"]], dtype=torch.long)
            b_hidden.append(model.encode_window(inp, mask, device)[0])
            b_offs.append(w["token_offsets"])

        all_hidden = cached_hidden[:b_start] + b_hidden + cached_hidden[b_end:]
        all_offs   = [w["token_offsets"] for w in windows]
        doc_h      = aggregate_embeddings(all_hidden, all_offs, max_tok, H, device)

        gs = torch.tensor([s for s, e in gold_spans_list],
                          dtype=torch.long, device=device).clamp(0, max_tok - 1)
        ge = torch.tensor([e for s, e in gold_spans_list],
                          dtype=torch.long, device=device).clamp(0, max_tok - 1)
        sv   = model.span_repr(doc_h, gs, ge)
        loss = model.mention_ranking_loss(sv, gold_spans_list, gold_clusters, device)
        (loss / n_blocks).backward()
        total_loss_val += loss.item()
        del b_hidden, all_hidden, doc_h, sv

    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    return total_loss_val / n_blocks


# Inference & Metrics
@torch.no_grad()
def predict_clusters(model, doc, tokenizer, device, max_tokens, stride, epsilon):
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
            if ner_map.get(si) == ner_map.get(sj):
                union(i, best_j)

    clusters_out: Dict[int, list] = defaultdict(list)
    for i in range(len(all_spans)):
        clusters_out[find(i)].append(all_spans[i])
    return [v for v in clusters_out.values() if len(v) >= 2]


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
    key, resp = _sets(gold), _sets(pred)
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
            gc = key[km[m]]; rc = resp[rm[m]] if m in rm else {m}
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
    sim    = np.array([[len(k & r) / len(k | r) if k | r else 0.0
                        for r in resp] for k in key])
    ri, ci = linear_sum_assignment(-sim)
    s      = sum(sim[r, c] for r, c in zip(ri, ci))
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


def evaluate(model, docs, tokenizer, device, max_tokens, stride, split="val"):
    model.eval()
    flat_gold, flat_pred = [], []
    epsilon = model.epsilon.item()
    for doc in tqdm(docs, desc=f"  Evaluating [{split}]", unit="doc", leave=False):
        dk = doc["doc_key"]
        flat_gold += prefix_clusters(doc["gold_clusters"], dk)
        flat_pred += prefix_clusters(
            predict_clusters(model, doc, tokenizer, device,
                             max_tokens, stride, epsilon), dk
        )
    muc   = muc_score(flat_pred, flat_gold)
    b3    = b3_score(flat_pred, flat_gold)
    ceafe = ceaf_e_score(flat_pred, flat_gold)
    lea   = lea_score(flat_pred, flat_gold)
    return {
        "MUC": muc, "B3": b3, "CEAF_e": ceafe,
        "CoNLL_F1": conll_f1(muc, b3, ceafe), "LEA": lea,
    }


def save_model(model, tokenizer, output_dir: Path, params: dict, extra: dict = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))
    cabecas = {k: v for k, v in model.state_dict().items()
               if not k.startswith("encoder.")}
    torch.save(cabecas, output_dir / "cabecas_coref.pt")
    coref_config = {
        "hidden_size":       model.hidden,
        "ffnn_dim":          params["ffnn_dim"],
        "max_span_width":    params["max_span_width"],
        "max_antecedents":   params["max_antecedents"],
        "max_cluster_size":  params["max_cluster_size"],
        "dropout":           params["dropout"],
        "max_tokens":        params["max_tokens"],
        "stride":            params["stride"],
        "dist_buckets":      DIST_BUCKETS,
        "ner_union_margin":  NER_UNION_MARGIN,
        "min_cluster_score": MIN_CLUSTER_SCORE,
        "epsilon":           model.epsilon.item(),
        "domain":            "atas municipais portuguesas",
        "language":          "pt",
    }
    if extra:
        coref_config.update(extra)
    with open(output_dir / "coref_config.json", "w", encoding="utf-8") as f:
        json.dump(coref_config, f, indent=2, ensure_ascii=False)


# Main
def main():
    parser = argparse.ArgumentParser(
        description="Train coreference model on Portuguese municipal minutes"
    )
    parser.add_argument("--config",     type=str, required=True,
                        help="Config name (e.g., spanbert_pt, bertimbau_coref)")
    parser.add_argument("--model_name", type=str,
                        help="Override model name from config")
    parser.add_argument("--json_dir",   type=str, default="data/coreference_dataset",
                        help="Directory with JSON document files")
    parser.add_argument("--output_dir", type=str,
                        help="Override output directory")
    args = parser.parse_args()

    # Load configuration
    config_path = "config/training_configs.json"
    with open(config_path, "r") as f:
        all_configs = json.load(f)

    if args.config not in all_configs:
        print(f"Error: Config '{args.config}' not found in {config_path}")
        return

    params = all_configs[args.config]
    if args.model_name:
        params["model_name"] = args.model_name

    json_dir    = Path(args.json_dir)
    output_dir  = Path(args.output_dir) if args.output_dir \
                  else Path(f"models/general_model/coref/{args.config}")
    results_dir = Path(f"results/coreference/general_model/{args.config}")
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Config: {args.config} | Model: {params['model_name']} | Device: {device}")

    # Load split definitions from split_info.json
    print("Loading documents...")
    train_files, val_files, test_files = load_splits(json_dir)
    train_docs = load_split(train_files, json_dir, "train")
    val_docs   = load_split(val_files,   json_dir, "val")
    test_docs  = load_split(test_files,  json_dir, "test")

    max_tokens = params["max_tokens"]
    stride     = params["stride"]

    # Initialise model
    model = CorefModel(
        model_name=params["model_name"],
        ffnn_dim=params["ffnn_dim"],
        max_span_width=params["max_span_width"],
        max_antecedents=params["max_antecedents"],
        max_cluster_size=params["max_cluster_size"],
        dropout=params["dropout"],
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(params["model_name"])

    # Optimizer with differential learning rate for encoder vs heads
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(),        "lr": params["learning_rate"] * params["encoder_lr_factor"]},
        {"params": model.mention_scorer.parameters(), "lr": params["learning_rate"]},
        {"params": model.ant_scorer.parameters(),     "lr": params["learning_rate"]},
        {"params": model.head_attn.parameters(),      "lr": params["learning_rate"]},
        {"params": model.pos_emb.parameters(),        "lr": params["learning_rate"]},
        {"params": model.dist_emb.parameters(),       "lr": params["learning_rate"]},
        {"params": model.width_emb.parameters(),      "lr": params["learning_rate"]},
        {"params": [model.epsilon],                   "lr": params["learning_rate"]},
    ], weight_decay=params["weight_decay"])

    total_steps = len(train_docs) * params["epochs"]
    warmup_steps = int(total_steps * params["warmup_ratio"])
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_f1 = -1.0
    best_epoch  = 0
    global_bar  = tqdm(total=params["epochs"] * len(train_docs),
                       desc="Training", unit="doc", dynamic_ncols=True)
    train_start = time.time()

    for epoch in range(1, params["epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        epoch_loss  = 0.0
        epoch_start = time.time()

        import random
        shuffled = train_docs.copy()
        random.shuffle(shuffled)

        for doc in shuffled:
            loss = train_one_doc(
                model, doc, tokenizer, optimizer, scheduler,
                device, max_tokens, stride, params["grad_acc"]
            )
            epoch_loss += loss
            elapsed  = time.time() - train_start
            done     = (epoch - 1) * len(train_docs) + shuffled.index(doc) + 1
            total    = params["epochs"] * len(train_docs)
            eta_secs = (elapsed / max(done, 1)) * (total - done)
            global_bar.set_postfix(
                epoch=f"{epoch}/{params['epochs']}",
                loss=f"{loss:.4f}", eta=fmt(eta_secs)
            )
            global_bar.update(1)

        avg_loss = epoch_loss / max(len(train_docs), 1)
        print(f"\nEpoch {epoch}/{params['epochs']} | "
              f"{fmt(time.time() - epoch_start)} | Loss={avg_loss:.4f} | "
              f"epsilon={model.epsilon.item():.4f}")

        val_m  = evaluate(model, val_docs, tokenizer, device,
                          max_tokens, stride, split="val")
        val_f1 = val_m["CoNLL_F1"]
        print(f"  Val CoNLL={val_f1:.4f} | MUC={val_m['MUC']['f1']:.4f} "
              f"B3={val_m['B3']['f1']:.4f} CEAF-e={val_m['CEAF_e']['f1']:.4f} "
              f"LEA={val_m['LEA']['f1']:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            save_model(model, tokenizer, output_dir, params,
                       extra={"best_val_conll": val_f1, "best_epoch": epoch})
            print(f"  ★ Best model saved | CoNLL={val_f1:.4f}")

        if best_epoch > 0 and (epoch - best_epoch) >= params["patience"]:
            print(f"Early stopping at epoch {epoch} "
                  f"(best: epoch {best_epoch}, CoNLL={best_val_f1:.4f})")
            break

    global_bar.close()
    print(f"\nTraining complete in {fmt(time.time() - train_start)}")
    print(f"Best epoch: {best_epoch} | Best val CoNLL: {best_val_f1:.4f}")

    # Evaluate on test set with best model
    print("Loading best model for final test evaluation...")
    from transformers import AutoModel as _AM
    best_model = CorefModel(
        model_name=str(output_dir),
        ffnn_dim=params["ffnn_dim"],
        max_span_width=params["max_span_width"],
        max_antecedents=params["max_antecedents"],
        max_cluster_size=params["max_cluster_size"],
        dropout=0.0,
    )
    cabecas = torch.load(output_dir / "cabecas_coref.pt",
                         map_location=device, weights_only=False)
    state   = best_model.state_dict()
    state.update({k: v for k, v in cabecas.items() if k in state})
    best_model.load_state_dict(state)
    best_model.to(device)

    test_m = evaluate(best_model, test_docs, tokenizer, device,
                      max_tokens, stride, split="test")
    print(f"\nTest Results:")
    print(f"  {'Metric':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*44}")
    for name, v in [("MUC", test_m["MUC"]), ("B3", test_m["B3"]),
                    ("CEAF-e", test_m["CEAF_e"]), ("LEA", test_m["LEA"])]:
        print(f"  {name:<12} {v['precision']:>10.4f} "
              f"{v['recall']:>10.4f} {v['f1']:>10.4f}")
    print(f"  {'CoNLL F1':<12} {'':>10} {'':>10} "
          f"{test_m['CoNLL_F1']:>10.4f}")

    results = {
        "config":            args.config,
        "model_name":        params["model_name"],
        "best_epoch":        best_epoch,
        "best_val_CoNLL_F1": best_val_f1,
        "test":              test_m,
        "params":            params,
    }
    out_path = results_dir / f"results_{args.config}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved in: {out_path}")
    print(f"Model saved in:   {output_dir}")


if __name__ == "__main__":
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    main()