import os
import json
import logging
import argparse
import random
import math
import time
from pathlib import Path
from typing import Iterator, Tuple
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, get_linear_schedule_with_warmup

try:
    from tqdm import tqdm
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm", "-q"])
    from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger("continued_pretraining")

# Default paths — override via command line arguments
DEFAULT_MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
DEFAULT_OUTPUT_DIR = "models/continued_pretraining/spanbert_pt"
DEFAULT_DATA_DIR   = "data/pretraining"

# Training hyperparameters
MAX_SEQ_LEN  = 512
MLM_PROB     = 0.15
BATCH_SIZE   = 8
GRAD_ACC     = 4
LR           = 5e-5
WARMUP_RATIO = 0.06
CHUNK_LINES  = 10_000

CKPT_FILE = "training_state.pt"


def fmt(s):
    return time.strftime("%H:%M:%S", time.gmtime(s))


def count_lines(path: Path) -> int:
    # Count lines efficiently without loading the full file into memory
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            n += chunk.count(b"\n")
    return n


def estimate_dataset_size(data_dir: Path, tokenizer) -> Tuple[int, int]:
    # Estimate number of sequences by sampling 5000 lines and extrapolating
    files        = list(data_dir.rglob("*.txt")) + list(data_dir.rglob("*.json"))
    total_lines  = 0
    sample_tokens = 0
    sample_lines  = 0
    SAMPLE = 5000

    for path in files:
        total_lines += count_lines(path)
        if sample_lines < SAMPLE:
            with open(path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line and len(line) > 20:
                        sample_tokens += len(tokenizer.encode(
                            line[:500], add_special_tokens=False))
                        sample_lines  += 1
                    if sample_lines >= SAMPLE:
                        break

    if sample_lines == 0:
        return 0, 0

    tokens_per_line  = sample_tokens / sample_lines
    total_tokens_est = int(tokens_per_line * total_lines)
    content_len      = MAX_SEQ_LEN - 2
    n_sequences_est  = total_tokens_est // content_len

    return total_lines, n_sequences_est


class StreamingMLMDataset(IterableDataset):
    # Streaming dataset that reads files line by line without loading into RAM
    # Accepts .txt files (line by line) and .json files (tokens/text field)

    def __init__(self, data_dir: Path, tokenizer,
                 max_seq_len: int = MAX_SEQ_LEN, seed: int = 42):
        self.data_dir    = data_dir
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len
        self.seed        = seed
        self.files       = (list(data_dir.rglob("*.txt")) +
                            list(data_dir.rglob("*.json")))
        if not self.files:
            raise ValueError(f"No files found in {data_dir}")
        logger.info(f"Streaming dataset: {len(self.files)} files found in {data_dir}")

    def _line_iterator(self) -> Iterator[str]:
        # Iterate over all lines from all files, shuffling file order each time
        files = self.files.copy()
        random.shuffle(files)
        for path in files:
            try:
                if path.suffix == ".txt":
                    with open(path, encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            line = line.strip()
                            if line and len(line) > 20:
                                yield line

                elif path.suffix == ".json":
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict) and "tokens" in data:
                        yield " ".join(str(t) for t in data["tokens"])
                    elif isinstance(data, dict):
                        for field in ["text", "content"]:
                            if field in data and len(data[field]) > 20:
                                yield data[field].strip()
                                break
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, str) and len(item) > 20:
                                yield item.strip()
                            elif isinstance(item, dict):
                                for field in ["text", "content", "tokens"]:
                                    if field in item:
                                        val  = item[field]
                                        text = (" ".join(str(t) for t in val)
                                                if isinstance(val, list) else str(val))
                                        if len(text) > 20:
                                            yield text
                                        break
            except Exception as e:
                logger.warning(f"Error reading {path.name}: {e}")

    def __iter__(self):
        cls_id      = self.tokenizer.cls_token_id
        sep_id      = self.tokenizer.sep_token_id
        pad_id      = self.tokenizer.pad_token_id
        content_len = self.max_seq_len - 2
        buffer      = []

        for line in self._line_iterator():
            # Tokenize the current line and add to the rolling buffer
            ids = self.tokenizer.encode(line[:2000], add_special_tokens=False)
            buffer.extend(ids)

            # Emit complete sequences as the buffer fills up
            while len(buffer) >= content_len:
                chunk     = buffer[:content_len]
                buffer    = buffer[content_len:]
                input_ids = [cls_id] + chunk + [sep_id]
                pad       = self.max_seq_len - len(input_ids)
                input_ids += [pad_id] * pad
                att_mask   = [1] * (content_len + 2) + [0] * pad
                yield {
                    "input_ids":      torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(att_mask,  dtype=torch.long),
                }

        # Emit remaining tokens in the buffer as the final sequence
        if len(buffer) >= 10:
            chunk     = buffer[:content_len]
            input_ids = [cls_id] + chunk + [sep_id]
            pad       = self.max_seq_len - len(input_ids)
            input_ids += [pad_id] * pad
            att_mask   = [1] * (len(chunk) + 2) + [0] * pad
            yield {
                "input_ids":      torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(att_mask,  dtype=torch.long),
            }


def collate_mlm(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
    }


def mask_tokens(input_ids: torch.Tensor, tokenizer, mlm_prob: float = MLM_PROB):
    # Standard BERT MLM masking: 80% [MASK], 10% random, 10% unchanged
    labels       = input_ids.clone()
    special_ids  = {tokenizer.cls_token_id, tokenizer.sep_token_id,
                    tokenizer.pad_token_id, tokenizer.unk_token_id}
    prob_matrix  = torch.full(input_ids.shape, mlm_prob)
    special_mask = torch.tensor(
        [[t in special_ids for t in row.tolist()] for row in input_ids],
        dtype=torch.bool)
    prob_matrix.masked_fill_(special_mask, 0.0)
    masked             = torch.bernoulli(prob_matrix).bool()
    labels[~masked]    = -100
    replace            = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked
    input_ids[replace] = tokenizer.mask_token_id
    rand_rep           = (torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool()
                          & masked & ~replace)
    rand_words         = torch.randint(len(tokenizer), input_ids.shape, dtype=torch.long)
    input_ids[rand_rep] = rand_words[rand_rep]
    return input_ids, labels


def save_checkpoint(output_dir, model, optimizer, scheduler,
                    epoch, global_step, best_loss, tokenizer):
    # Save model weights, tokenizer and training state for resuming
    ckpt_dir = Path(output_dir) / f"checkpoint-epoch-{epoch}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))
    torch.save({
        "epoch": epoch, "global_step": global_step,
        "best_loss": best_loss,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, ckpt_dir / CKPT_FILE)
    logger.info(f"Checkpoint saved: {ckpt_dir}")


def find_last_checkpoint(output_dir: Path):
    # Find the most recent valid checkpoint to resume training from
    ckpts = sorted(Path(output_dir).glob("checkpoint-epoch-*"),
                   key=lambda p: int(p.name.split("-")[-1]))
    for ckpt in reversed(ckpts):
        if (ckpt / CKPT_FILE).exists():
            return ckpt
    return None


def estimate_time(data_dir, tokenizer, model, device,
                  batch_size, grad_acc, epochs, n_test=20):
    # Benchmark n_test batches and project total training time
    logger.info("=" * 58)
    logger.info("TIME ESTIMATE")
    logger.info("=" * 58)

    logger.info("Estimating dataset size (sampling 5000 lines)...")
    total_lines, n_seqs_est = estimate_dataset_size(Path(data_dir), tokenizer)
    batches_per_epoch = n_seqs_est // batch_size
    steps_per_epoch   = batches_per_epoch // grad_acc

    logger.info(f"  Estimated total lines:     {total_lines:,}")
    logger.info(f"  Estimated sequences:       {n_seqs_est:,}")
    logger.info(f"  Estimated batches/epoch:   {batches_per_epoch:,}")
    logger.info(f"  Estimated steps/epoch:     {steps_per_epoch:,}")

    logger.info(f"Benchmarking {n_test} real batches...")
    dataset    = StreamingMLMDataset(Path(data_dir), tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            collate_fn=collate_mlm, num_workers=0)

    model.train()
    times, losses = [], []

    for i, batch in enumerate(dataloader):
        if i >= 2: break
        ids, lbl = mask_tokens(batch["input_ids"].clone(), tokenizer)
        model(input_ids=ids.to(device),
              attention_mask=batch["attention_mask"].to(device),
              labels=lbl.to(device))

    if device.type == "cuda":
        torch.cuda.synchronize()

    for i, batch in enumerate(dataloader):
        if i >= n_test: break
        t0 = time.perf_counter()
        ids, lbl = mask_tokens(batch["input_ids"].clone(), tokenizer)
        out = model(input_ids=ids.to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=lbl.to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        losses.append(out.loss.item())

    sec_per_batch = sum(times) / len(times)
    sec_per_step  = sec_per_batch * grad_acc
    sec_per_epoch = sec_per_step * steps_per_epoch
    total_secs    = sec_per_epoch * epochs
    ppl_init      = math.exp(min(sum(losses) / len(losses), 20))

    logger.info(f"  sec/batch:   {sec_per_batch:.3f}s")
    logger.info(f"  sec/step:    {sec_per_step:.3f}s")
    logger.info(f"  initial PPL: {ppl_init:.1f}")
    logger.info(f"  time/epoch:  {fmt(sec_per_epoch)}")
    logger.info(f"  TOTAL ({epochs} epochs): {fmt(total_secs)}")
    logger.info("=" * 58)

    return steps_per_epoch


def main():
    parser = argparse.ArgumentParser(
        description="Continued Pre-Training for Portuguese coreference (SpanBERT-PT)"
    )
    parser.add_argument("--model_name",    default=DEFAULT_MODEL_NAME,
                        help="Base model name or path (HuggingFace or local)")
    parser.add_argument("--data_dir",      default=DEFAULT_DATA_DIR,
                        help="Directory with .txt or .json pretraining files")
    parser.add_argument("--output_dir",    default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for the pretrained model")
    parser.add_argument("--epochs",        default=6,          type=int)
    parser.add_argument("--batch_size",    default=BATCH_SIZE, type=int)
    parser.add_argument("--grad_acc",      default=GRAD_ACC,   type=int)
    parser.add_argument("--lr",            default=LR,         type=float)
    parser.add_argument("--max_seq_len",   default=MAX_SEQ_LEN, type=int)
    parser.add_argument("--mlm_prob",      default=MLM_PROB,   type=float)
    parser.add_argument("--seed",          default=42,          type=int)
    parser.add_argument("--resume",        action="store_true",
                        help="Resume from the last checkpoint in --output_dir")
    parser.add_argument("--estimate_only", action="store_true",
                        help="Only estimate training time, do not train")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    data_dir   = Path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Device: {device}")
    logger.info(f"Base model: {args.model_name}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Load model and tokenizer — from checkpoint if resuming, from base model otherwise
    start_epoch = 1
    global_step = 0
    best_loss   = float("inf")
    last_ckpt   = find_last_checkpoint(output_dir) if args.resume else None

    if last_ckpt:
        logger.info(f"Resuming from checkpoint: {last_ckpt}")
        tokenizer = AutoTokenizer.from_pretrained(str(last_ckpt))
        model     = AutoModelForMaskedLM.from_pretrained(str(last_ckpt)).to(device)
        state     = torch.load(last_ckpt / CKPT_FILE, map_location=device)
        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        best_loss   = state["best_loss"]
        logger.info(f"Resuming from epoch {start_epoch}, step {global_step}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model     = AutoModelForMaskedLM.from_pretrained(args.model_name).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Estimate training time before starting
    steps_per_epoch = estimate_time(
        data_dir, tokenizer, model, device,
        args.batch_size, args.grad_acc, args.epochs
    )

    if args.estimate_only:
        logger.info("--estimate_only mode: exiting without training.")
        return

    # Setup optimizer with linear warmup scheduler
    total_steps  = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=0.01, betas=(0.9, 0.999), eps=1e-6
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    if last_ckpt and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])

    global_bar  = tqdm(total=total_steps, initial=global_step,
                       desc="Continued Pre-Training", unit="step",
                       dynamic_ncols=True, position=0)
    train_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Create a new dataset instance per epoch to reshuffle file order
        dataset    = StreamingMLMDataset(data_dir, tokenizer,
                                         seed=args.seed + epoch)
        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                collate_fn=collate_mlm, num_workers=0)

        epoch_loss  = 0.0
        epoch_steps = 0
        epoch_start = time.time()

        epoch_bar = tqdm(desc=f"Epoch {epoch}/{args.epochs}",
                         total=steps_per_epoch, unit="step",
                         leave=False, dynamic_ncols=True, position=1)

        for step, batch in enumerate(dataloader):
            ids, labels = mask_tokens(
                batch["input_ids"].clone(), tokenizer, args.mlm_prob
            )
            out = model(
                input_ids=ids.to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=labels.to(device)
            )
            (out.loss / args.grad_acc).backward()
            epoch_loss += out.loss.item()

            if (step + 1) % args.grad_acc == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                epoch_steps += 1

                ppl        = math.exp(min(out.loss.item(), 20))
                elapsed    = time.time() - train_start
                eta_total  = (elapsed / max(global_step, 1)) * (total_steps - global_step)
                ep_elapsed = time.time() - epoch_start
                eta_epoch  = (ep_elapsed / max(epoch_steps, 1)) * (steps_per_epoch - epoch_steps)

                epoch_bar.set_postfix(
                    loss=f"{out.loss.item():.3f}", ppl=f"{ppl:.1f}",
                    eta_ep=fmt(eta_epoch), refresh=False
                )
                global_bar.set_postfix(
                    ep=f"{epoch}/{args.epochs}", loss=f"{out.loss.item():.3f}",
                    ppl=f"{ppl:.1f}", eta=fmt(eta_total), refresh=False
                )
                global_bar.update(1)
                epoch_bar.update(1)

            torch.cuda.empty_cache()

        epoch_bar.close()
        avg_loss = epoch_loss / max(step + 1, 1)
        avg_ppl  = math.exp(min(avg_loss, 20))
        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"{fmt(time.time() - epoch_start)} | "
            f"Loss={avg_loss:.4f} | PPL={avg_ppl:.2f}"
        )

        # Save checkpoint after each epoch to allow resuming
        save_checkpoint(output_dir, model, optimizer, scheduler,
                        epoch, global_step, best_loss, tokenizer)

        # Save the best model separately
        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
            logger.info(
                f"Best model saved | Loss={best_loss:.4f} | PPL={avg_ppl:.2f}"
            )

    global_bar.close()
    logger.info(f"Pre-training complete in {fmt(time.time() - train_start)}")
    logger.info(f"Best loss: {best_loss:.4f} | PPL: {math.exp(min(best_loss, 20)):.2f}")
    logger.info(f"Model saved in: {output_dir}")
    logger.info(
        f"Next step — fine-tune for coreference:\n"
        f"  python src/coref/supervised/general_model/run_pipeline.py "
        f"--model_name {output_dir}"
    )


if __name__ == "__main__":
    main()