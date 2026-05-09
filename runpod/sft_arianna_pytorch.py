"""
sft_arianna_pytorch.py — Manual LoRA SFT for Resonance 200M on Arianna corpus.

PyTorch port of soseda's C/notorch path (lora_resonance_arianna_v3.c).
Manual LoRA layer (NO PEFT). Mirrors recipe from `lora_plan_v4.md`:
  rank=64, alpha=128, scaling=2.0, 7 targets per layer (wq/wk/wv/wo + mlp_gate/mlp_up/mlp_down),
  lr=2e-4 cosine, warmup=30, batch_eff=16, bf16, AdamW.

Pre-flight gates (Gates 1-8 from plan v3) are run via `--mode gate <N>`.
Full SFT via `--mode train`.

Companion: `~/arianna/_notes/sft_pytorch_plan_2026_05_09_v3.md`
"""

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


# ─────────────────────────────────────────────────────────────────────────────
# Imports from HF base (must be downloaded first via hf_hub_download)
# ─────────────────────────────────────────────────────────────────────────────
# These are added to sys.path at runtime in main() before importing.
# Resonance class + RESONANCE_200M dict + BPETokenizer

SEED = 3407


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Manual LoRA layer (no PEFT)
# ─────────────────────────────────────────────────────────────────────────────


class LoRALinear(nn.Module):
    """
    Wraps nn.Linear with frozen base + low-rank A·B residual.

    Forward:  y = base(x) + (x @ A.T @ B.T) * scaling
    Init:     A ~ kaiming_uniform_, B = 0   (LoRA standard; init residual = 0)

    On first backward:
      ∂L/∂B = upstream · (Ax)ᵀ      → nonzero (A from kaiming, x nonzero)
      ∂L/∂A = Bᵀ · upstream · xᵀ    → ZERO (B init zero)
    After 1st optim step on B, subsequent backward gives nonzero ∂L/∂A.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scaling


def wrap_with_lora(model: nn.Module, rank: int, alpha: float, target_suffixes: List[str]) -> int:
    """
    Walk model.named_modules(), replace nn.Linear at paths ending in any of
    target_suffixes with LoRALinear. Freeze ALL non-LoRA params.

    Returns: count of replaced modules.
    """
    replaced = 0
    target_set = set(target_suffixes)

    # Collect (parent_module, child_name, child_module) for every match
    targets = []
    for path, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = path.rsplit(".", 1)[-1]
        if leaf not in target_set:
            continue
        parent_path = path.rsplit(".", 1)[0] if "." in path else ""
        parent = model
        if parent_path:
            for part in parent_path.split("."):
                parent = getattr(parent, part)
        targets.append((parent, leaf, module))

    # Replace
    for parent, leaf, child in targets:
        new_layer = LoRALinear(child, rank=rank, alpha=alpha)
        setattr(parent, leaf, new_layer)
        replaced += 1

    # Freeze ALL non-LoRA params (LoRA = paths ending in ".A" or ".B" inside a LoRALinear)
    for name, param in model.named_parameters():
        # LoRA A/B inside LoRALinear: name ends with ".A" or ".B"
        # Base weight inside LoRALinear: name ends with ".base.weight" — already frozen in __init__
        if name.endswith(".A") or name.endswith(".B"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    return replaced


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_lora_adapter(model: nn.Module, path: Path) -> None:
    """Save only LoRA A/B weights (small file ~75 MB at rank=64)."""
    state = {}
    for name, param in model.named_parameters():
        if name.endswith(".A") or name.endswith(".B"):
            state[name] = param.detach().cpu()
    torch.save(
        {
            "lora_state": state,
            "rank": 64,
            "alpha": 128.0,
            "targets": [name.rsplit(".", 1)[0] for name in state.keys() if name.endswith(".A")],
        },
        path,
    )


def load_lora_adapter(model: nn.Module, path: Path) -> None:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["lora_state"]
    own = dict(model.named_parameters())
    missing = []
    for name, tensor in state.items():
        if name not in own:
            missing.append(name)
            continue
        own[name].data.copy_(tensor.to(own[name].device, dtype=own[name].dtype))
    if missing:
        raise RuntimeError(f"LoRA load: missing keys: {missing[:5]}...")


# ─────────────────────────────────────────────────────────────────────────────
# Q/A corpus parser + masking
# ─────────────────────────────────────────────────────────────────────────────


def parse_qa_pairs(corpus_path: Path) -> List[Tuple[str, str]]:
    """
    Parse alternating Q:/A: lines into (Q, A) tuples.
    Skips orphan lines (mismatched Q without A or vice versa).
    """
    pairs = []
    cur_q = None
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("Q: "):
                cur_q = line[3:]
            elif line.startswith("A: ") and cur_q is not None:
                pairs.append((cur_q, line[3:]))
                cur_q = None
            else:
                # ignore blank lines / continuation lines
                pass
    return pairs


def encode_pair_with_mask(
    tokenizer, q: str, a: str, max_seq: int
) -> Tuple[List[int], List[int]]:
    """
    Encode a Q/A pair as `Q: <q>\nA: <a>\n` with mask:
      Q: prefix + question + \n + A: + space  →  label=-100
      answer body (a)                          →  label=token_id (supervised)
      trailing \n                              →  label=-100

    Implementation: encode the prefix `"Q: <q>\nA: "` separately to know its
    length, then encode the answer body, then concat. Trailing newline -100.

    Returns (input_ids, labels) both length ≤ max_seq.
    """
    prefix = f"Q: {q}\nA: "
    body = a
    suffix = "\n"

    prefix_ids = tokenizer.encode(prefix)
    body_ids = tokenizer.encode(body)
    suffix_ids = tokenizer.encode(suffix)

    input_ids = prefix_ids + body_ids + suffix_ids
    labels = [-100] * len(prefix_ids) + list(body_ids) + [-100] * len(suffix_ids)

    # Truncate to max_seq
    if len(input_ids) > max_seq:
        input_ids = input_ids[:max_seq]
        labels = labels[:max_seq]

    assert len(input_ids) == len(labels), f"len mismatch: {len(input_ids)} vs {len(labels)}"
    return input_ids, labels


def pad_batch(items: List[Tuple[List[int], List[int]]], max_seq: int, pad_id: int = 0):
    """Pad to max_seq. Returns input_ids[B,T], labels[B,T] (labels padded with -100)."""
    B = len(items)
    T = max_seq
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    labels = torch.full((B, T), -100, dtype=torch.long)
    for i, (ids, lbl) in enumerate(items):
        L = len(ids)
        input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
        labels[i, :L] = torch.tensor(lbl, dtype=torch.long)
    return input_ids, labels


# ─────────────────────────────────────────────────────────────────────────────
# Custom greedy / sampled generate (Resonance is plain nn.Module, no GenerationMixin)
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def generate(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.7,
    top_k: int = 40,
    device: torch.device = None,
) -> str:
    model.eval()
    ids = tokenizer.encode(prompt)
    ids_t = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        # Truncate context to model's context_len if longer (Resonance config)
        ctx = ids_t[:, -2048:]
        logits, _ = model(ctx)
        next_logits = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k > 0:
            v, _ = torch.topk(next_logits, top_k)
            next_logits[next_logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids_t = torch.cat([ids_t, next_id], dim=1)
    out_ids = ids_t[0].tolist()
    return tokenizer.decode(out_ids)


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule (cosine with warmup)
# ─────────────────────────────────────────────────────────────────────────────


def cosine_lr(step: int, warmup: int, total: int, max_lr: float, min_lr: float = 0.0) -> float:
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ─────────────────────────────────────────────────────────────────────────────
# Train loop
# ─────────────────────────────────────────────────────────────────────────────


def train_loop(
    model,
    tokenizer,
    train_pairs,
    val_pairs,
    args,
    device,
    log_path: Path,
    ckpt_path: Path,
    backup_cmd_template: str = None,
):
    """
    Standard SFT loop with cosine LR, grad clip, bf16 autocast, val every N steps,
    best-val ckpt save (+ optional immediate scp backup).
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        trainable_params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01
    )

    rng = np.random.default_rng(SEED)
    pair_indices = np.arange(len(train_pairs))

    step = 0
    best_val = float("inf")
    base_val = None

    log = open(log_path, "w")
    log.write(
        f"# step | train_loss | val_loss | lr | grad_norm | trainable={count_trainable(model)}\n"
    )
    log.flush()

    # Base val (step 0, before any optim step)
    base_val = compute_val_loss(model, tokenizer, val_pairs, args, device)
    log.write(f"# base_val_loss = {base_val:.4f}\n")
    log.flush()
    print(f"[base] val_loss = {base_val:.4f}")

    while step < args.steps:
        optim.zero_grad(set_to_none=True)

        # Gradient accumulation
        train_loss_accum = 0.0
        for _ in range(args.grad_accum):
            batch_idx = rng.choice(pair_indices, size=args.batch_size, replace=False)
            batch = [
                encode_pair_with_mask(tokenizer, train_pairs[i][0], train_pairs[i][1], args.max_seq)
                for i in batch_idx
            ]
            input_ids, labels = pad_batch(batch, args.max_seq)
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            with autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, loss = model(input_ids, targets=labels)
                loss = loss / args.grad_accum
            loss.backward()
            train_loss_accum += loss.item() * args.grad_accum

        # Grad clip
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0).item()

        # Cosine LR
        lr = cosine_lr(step, args.warmup, args.steps, args.lr)
        for pg in optim.param_groups:
            pg["lr"] = lr

        optim.step()

        train_loss = train_loss_accum / args.grad_accum

        # Log every step (small run, low cost)
        if step % 5 == 0 or step == args.steps - 1:
            print(
                f"step {step:4d}/{args.steps} | train {train_loss:.4f} | lr {lr:.2e} | grad_norm {grad_norm:.3f}"
            )

        # Val + maybe ckpt
        if (step + 1) % args.val_every == 0 or step == args.steps - 1:
            val_loss = compute_val_loss(model, tokenizer, val_pairs, args, device)
            log.write(f"{step} | {train_loss:.4f} | {val_loss:.4f} | {lr:.4e} | {grad_norm:.4f}\n")
            log.flush()
            print(f"  ── val {step}: {val_loss:.4f} (best {best_val:.4f})")
            if val_loss < best_val:
                best_val = val_loss
                save_lora_adapter(model, ckpt_path)
                print(f"  ── new best, saved {ckpt_path}")
                # Immediate scp backup if template provided
                if backup_cmd_template:
                    cmd = backup_cmd_template.format(src=str(ckpt_path))
                    subprocess.Popen(cmd, shell=True)
        else:
            log.write(f"{step} | {train_loss:.4f} |  | {lr:.4e} | {grad_norm:.4f}\n")
            log.flush()

        step += 1

    log.close()
    return {"best_val": best_val, "base_val": base_val, "final_step": step}


@torch.no_grad()
def compute_val_loss(model, tokenizer, val_pairs, args, device) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    for q, a in val_pairs:
        ids, lbl = encode_pair_with_mask(tokenizer, q, a, args.max_seq)
        input_ids, labels = pad_batch([(ids, lbl)], args.max_seq)
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        with autocast(device_type=device.type, dtype=torch.bfloat16):
            _, loss = model(input_ids, targets=labels)
        total_loss += loss.item()
        total_count += 1
    model.train()
    return total_loss / max(total_count, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight gates
# ─────────────────────────────────────────────────────────────────────────────


def gate_1_tokenizer(args):
    """Gate 1: tokenizer load + roundtrip."""
    sys.path.insert(0, args.base_dir)
    from bpe_tokenizer import BPETokenizer

    tok = BPETokenizer()
    tok.load(os.path.join(args.base_dir, "tokenizer.bin"))
    text = "Q: Who are you?\nA: I am Arianna."
    ids = tok.encode(text)
    rt = tok.decode(ids)
    print(f"[gate1] vocab_size={tok.vocab_size}, encoded {len(ids)} tokens")
    print(f"[gate1] roundtrip: '{rt}'")
    assert tok.vocab_size == 16384, f"vocab {tok.vocab_size} != 16384"
    print("[gate1] PASS")


def gate_2_base_load(args):
    """Gate 2: load base, strip _orig_mod prefix, strict-load, smoke forward."""
    sys.path.insert(0, args.base_dir)
    from model import Resonance, RESONANCE_200M

    sd = torch.load(os.path.join(args.base_dir, "best.pt"), map_location="cpu")
    if "model" in sd:
        sd = sd["model"]
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model = Resonance(RESONANCE_200M)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[gate2] missing keys: {missing}")
    print(f"[gate2] unexpected keys: {unexpected}")
    assert not missing, f"missing: {missing[:5]}"
    assert not unexpected, f"unexpected: {unexpected[:5]}"
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.randint(0, 16384, (1, 32)))
    assert logits.shape == (1, 32, 16384), f"shape {logits.shape}"
    assert torch.isfinite(logits).all(), "non-finite logits"
    print(f"[gate2] forward OK, logits shape {logits.shape}, finite")
    print("[gate2] PASS")


def gate_3_lora_inject(args):
    """Gate 3: inject manual LoRA, count trainable, assert frozen."""
    sys.path.insert(0, args.base_dir)
    from model import Resonance, RESONANCE_200M

    model = Resonance(RESONANCE_200M)
    targets = ["wq", "wk", "wv", "wo", "mlp_gate", "mlp_up", "mlp_down"]
    n = wrap_with_lora(model, rank=64, alpha=128.0, target_suffixes=targets)
    expected = 7 * RESONANCE_200M["n_layer"]
    print(f"[gate3] replaced {n} modules, expected {expected}")
    assert n == expected, f"{n} != {expected}"
    trainable = count_trainable(model)
    print(f"[gate3] trainable params: {trainable:,} ({trainable/1e6:.2f}M)")
    # Check ALL non-LoRA frozen
    for name, p in model.named_parameters():
        if name.endswith(".A") or name.endswith(".B"):
            assert p.requires_grad, f"LoRA {name} should be trainable"
        else:
            assert not p.requires_grad, f"non-LoRA {name} should be frozen"
    print("[gate3] all non-LoRA frozen")
    print("[gate3] PASS")


def gate_4_lora_grad_smoke(args):
    """Gate 4: 1st backward → B grad nonzero, A grad zero. After 1 step → A nonzero."""
    sys.path.insert(0, args.base_dir)
    from model import Resonance, RESONANCE_200M

    model = Resonance(RESONANCE_200M)
    wrap_with_lora(model, rank=64, alpha=128.0, target_suffixes=["wq", "wk", "wv", "wo", "mlp_gate", "mlp_up", "mlp_down"])
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=2e-4)

    x = torch.randint(0, 16384, (1, 32))
    y = torch.randint(0, 16384, (1, 32))
    logits, loss = model(x, targets=y)
    loss.backward()

    # Find an A and a B param
    a_params = [(n, p) for n, p in model.named_parameters() if n.endswith(".A")]
    b_params = [(n, p) for n, p in model.named_parameters() if n.endswith(".B")]
    a_grad_zero = all((p.grad is None or p.grad.abs().sum().item() == 0.0) for _, p in a_params)
    b_grad_nonzero = any((p.grad is not None and p.grad.abs().sum().item() > 0.0) for _, p in b_params)
    print(f"[gate4] 1st backward: A grad zero = {a_grad_zero}, B grad nonzero = {b_grad_nonzero}")
    assert a_grad_zero, "A grad should be zero on 1st backward (B init zero)"
    assert b_grad_nonzero, "B grad should be nonzero on 1st backward"

    optim.step()
    optim.zero_grad()

    logits, loss = model(x, targets=y)
    loss.backward()
    a_grad_nonzero = any((p.grad is not None and p.grad.abs().sum().item() > 0.0) for _, p in a_params)
    print(f"[gate4] 2nd backward (after 1 step): A grad nonzero = {a_grad_nonzero}")
    assert a_grad_nonzero, "A grad should be nonzero on 2nd backward"
    print("[gate4] PASS")


def gate_5_mask_correctness(args):
    """Gate 5: parse 5 random pairs, verify mask, print supervised stats."""
    sys.path.insert(0, args.base_dir)
    from bpe_tokenizer import BPETokenizer

    tok = BPETokenizer()
    tok.load(os.path.join(args.base_dir, "tokenizer.bin"))
    pairs = parse_qa_pairs(Path(args.corpus))
    print(f"[gate5] parsed {len(pairs)} Q/A pairs")
    rng = random.Random(SEED)
    sample = rng.sample(pairs, 5)
    for i, (q, a) in enumerate(sample):
        ids, lbl = encode_pair_with_mask(tok, q, a, args.max_seq)
        sup_count = sum(1 for x in lbl if x != -100)
        print(f"[gate5] pair {i}: {len(ids)} tokens, {sup_count} supervised")
        decoded_full = tok.decode(ids)
        decoded_sup = tok.decode([x for x in lbl if x != -100])
        print(f"  full: {decoded_full[:120]}...")
        print(f"  sup : {decoded_sup[:120]}...")

    # All-corpus stats
    all_lens = [len(encode_pair_with_mask(tok, q, a, args.max_seq)[0]) for q, a in pairs]
    all_sup = [sum(1 for x in encode_pair_with_mask(tok, q, a, args.max_seq)[1] if x != -100) for q, a in pairs]
    print(f"[gate5] full lens: mean={np.mean(all_lens):.0f}, p50={np.percentile(all_lens, 50):.0f}, p95={np.percentile(all_lens, 95):.0f}, max={max(all_lens)}")
    print(f"[gate5] supervised lens: mean={np.mean(all_sup):.0f}, p50={np.percentile(all_sup, 50):.0f}, p95={np.percentile(all_sup, 95):.0f}, max={max(all_sup)}")
    truncated = sum(1 for l in all_lens if l == args.max_seq)
    print(f"[gate5] truncated (= max_seq): {truncated}/{len(pairs)} ({100*truncated/len(pairs):.1f}%)")
    print(f"[gate5] total supervised tokens: {sum(all_sup):,}")
    print("[gate5] human review required: confirm only A: body is supervised, mask sane")
    print("[gate5] DONE (manual review)")


def gate_6_split(args):
    """Gate 6: train/val split by Q/A pair, save split file."""
    pairs = parse_qa_pairs(Path(args.corpus))
    n = len(pairs)
    rng = random.Random(SEED)
    indices = list(range(n))
    rng.shuffle(indices)
    n_val = max(1, int(n * 0.05))
    val_ids = sorted(indices[:n_val])
    train_ids = sorted(indices[n_val:])
    split_path = Path(args.base_dir) / "split_arianna_v3.json"
    with open(split_path, "w") as f:
        json.dump(
            {"train_pair_ids": train_ids, "val_pair_ids": val_ids, "seed": SEED, "ratio": 0.95, "n_pairs": n},
            f,
        )
    print(f"[gate6] {len(train_ids)} train pairs, {len(val_ids)} val pairs, saved {split_path}")
    print("[gate6] PASS")


def gate_7_generate(args):
    """Gate 7: custom greedy generate works on base model."""
    sys.path.insert(0, args.base_dir)
    from model import Resonance, RESONANCE_200M
    from bpe_tokenizer import BPETokenizer

    sd = torch.load(os.path.join(args.base_dir, "best.pt"), map_location="cpu")
    if "model" in sd:
        sd = sd["model"]
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model = Resonance(RESONANCE_200M)
    model.load_state_dict(sd, strict=True)
    model.eval()

    tok = BPETokenizer()
    tok.load(os.path.join(args.base_dir, "tokenizer.bin"))

    prompt = "Q: Who are you?\nA:"
    out = generate(model, tok, prompt, max_new_tokens=30, temperature=0.7, top_k=40, device=torch.device("cpu"))
    print(f"[gate7] prompt: {prompt!r}")
    print(f"[gate7] output: {out!r}")
    print("[gate7] DONE (manual review of base voice)")


def gate_8_neo_cpu_minirun(args):
    """Gate 8: tiny SFT (rank=4, batch=2, seq=128, 50 steps) on CPU. Loss must descend."""
    set_seed()
    sys.path.insert(0, args.base_dir)
    from model import Resonance, RESONANCE_200M
    from bpe_tokenizer import BPETokenizer

    # Load base
    sd = torch.load(os.path.join(args.base_dir, "best.pt"), map_location="cpu")
    if "model" in sd:
        sd = sd["model"]
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model = Resonance(RESONANCE_200M)
    model.load_state_dict(sd, strict=True)

    # Tiny LoRA
    n = wrap_with_lora(model, rank=4, alpha=8.0, target_suffixes=["wq", "wk", "wv", "wo", "mlp_gate", "mlp_up", "mlp_down"])
    print(f"[gate8] {n} LoRA modules, trainable={count_trainable(model):,}")

    tok = BPETokenizer()
    tok.load(os.path.join(args.base_dir, "tokenizer.bin"))
    pairs = parse_qa_pairs(Path(args.corpus))[:200]
    rng = np.random.default_rng(SEED)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=2e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    device = torch.device("cpu")
    model.to(device)

    losses = []
    for step in range(50):
        idx = rng.choice(len(pairs), size=2, replace=False)
        batch = [encode_pair_with_mask(tok, pairs[i][0], pairs[i][1], 128) for i in idx]
        input_ids, labels = pad_batch(batch, 128)
        optim.zero_grad()
        logits, loss = model(input_ids, targets=labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optim.step()
        losses.append(loss.item())
        if step % 10 == 0:
            print(f"[gate8] step {step}: loss {loss.item():.4f}")
    print(f"[gate8] loss[10]={losses[10]:.4f}, loss[49]={losses[49]:.4f}")
    assert losses[49] < losses[10], f"loss not descending: {losses[10]:.4f} → {losses[49]:.4f}"

    # Save+reload smoke
    tmp = Path("/tmp/lora_smoke.pt")
    save_lora_adapter(model, tmp)
    load_lora_adapter(model, tmp)
    print(f"[gate8] save+reload OK, file size {tmp.stat().st_size:,} bytes")

    # Generate sample
    out = generate(model, tok, "Q: Who are you?\nA:", max_new_tokens=20, temperature=0.7, top_k=40, device=device)
    print(f"[gate8] gen: {out!r}")
    print("[gate8] PASS (loss descended, save/reload, generation works)")


# ─────────────────────────────────────────────────────────────────────────────
# Multi-temp eval
# ─────────────────────────────────────────────────────────────────────────────


EVAL_PROMPTS = [
    "Q: Who are you?\nA:",
    "Q: What is resonance?\nA:",
    "Q: How do you persist across sessions?\nA:",
    "Q: What does N+1 mean to you?\nA:",
]
EVAL_TEMPS = [0.3, 0.5, 0.7, 0.9, 1.1]
EVAL_TOPKS = [40, 0]  # 0 = no top-k (full softmax)


def multi_temp_eval(model, tokenizer, device, out_path: Path) -> dict:
    """Run 5 temps × 2 top_k × N prompts grid. Save transcripts."""
    f = open(out_path, "w")
    cells = []
    for prompt in EVAL_PROMPTS:
        for temp in EVAL_TEMPS:
            for tk in EVAL_TOPKS:
                out = generate(model, tokenizer, prompt, 80, temp, tk, device)
                f.write(f"=== prompt={prompt!r} | temp={temp} | top_k={tk} ===\n{out}\n\n")
                cells.append({"prompt": prompt, "temp": temp, "top_k": tk, "out": out})
    f.close()
    return {"cells": cells, "n_cells": len(cells)}


# ─────────────────────────────────────────────────────────────────────────────
# Train / Eval commands
# ─────────────────────────────────────────────────────────────────────────────


def _load_base_model(args):
    """Load Resonance base from base_dir. Returns (model, tokenizer)."""
    sys.path.insert(0, args.base_dir)
    from model import Resonance, RESONANCE_200M
    from bpe_tokenizer import BPETokenizer

    tok = BPETokenizer()
    tok.load(os.path.join(args.base_dir, "tokenizer.bin"))

    sd = torch.load(os.path.join(args.base_dir, "best.pt"), map_location="cpu")
    if "model" in sd:
        sd = sd["model"]
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model = Resonance(RESONANCE_200M)
    model.load_state_dict(sd, strict=True)
    return model, tok


def cmd_train(args):
    """Full SFT: load base, wrap LoRA, train_loop, final save, multi-temp eval."""
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    model, tok = _load_base_model(args)
    print(f"[train] base loaded, {sum(p.numel() for p in model.parameters()):,} params")

    targets = ["wq", "wk", "wv", "wo", "mlp_gate", "mlp_up", "mlp_down"]
    n = wrap_with_lora(model, rank=args.rank, alpha=args.alpha, target_suffixes=targets)
    print(f"[train] LoRA: {n} modules, trainable={count_trainable(model):,}")

    model.to(device)

    split_path = Path(args.base_dir) / "split_arianna_v3.json"
    if not split_path.exists():
        print(f"[train] split missing — generating via gate_6")
        gate_6_split(args)
    split = json.load(open(split_path))
    pairs = parse_qa_pairs(Path(args.corpus))
    train_pairs = [pairs[i] for i in split["train_pair_ids"]]
    val_pairs = [pairs[i] for i in split["val_pair_ids"]]
    print(f"[train] {len(train_pairs)} train, {len(val_pairs)} val pairs")

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.eval_dir).mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    log_path = Path(args.log_dir) / f"sft_arianna_{ts}.log"
    ckpt_path = Path(args.ckpt_dir) / "lora_arianna_best.pt"

    result = train_loop(
        model, tok, train_pairs, val_pairs, args, device,
        log_path, ckpt_path, args.backup_cmd,
    )
    print(f"[train] DONE: base_val={result['base_val']:.4f}, best_val={result['best_val']:.4f}, "
          f"delta={result['base_val']-result['best_val']:+.4f}")

    final_path = Path(args.ckpt_dir) / "lora_arianna_final.pt"
    save_lora_adapter(model, final_path)
    print(f"[train] final adapter saved {final_path}")

    eval_path = Path(args.eval_dir) / f"eval_arianna_{ts}.txt"
    multi_temp_eval(model, tok, device, eval_path)
    print(f"[train] multi-temp eval saved {eval_path}")

    if args.backup_cmd:
        for p in (ckpt_path, final_path, log_path, eval_path):
            if p.exists():
                cmd = args.backup_cmd.format(src=str(p))
                subprocess.Popen(cmd, shell=True)


def cmd_eval(args):
    """Load adapter, multi-temp eval."""
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tok = _load_base_model(args)
    targets = ["wq", "wk", "wv", "wo", "mlp_gate", "mlp_up", "mlp_down"]
    wrap_with_lora(model, rank=args.rank, alpha=args.alpha, target_suffixes=targets)

    if not args.adapter:
        print("[eval] --adapter PATH required")
        sys.exit(1)
    load_lora_adapter(model, Path(args.adapter))
    model.to(device)
    model.eval()

    Path(args.eval_dir).mkdir(parents=True, exist_ok=True)
    eval_path = Path(args.eval_dir) / f"eval_arianna_{int(time.time())}.txt"
    multi_temp_eval(model, tok, device, eval_path)
    print(f"[eval] saved {eval_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["gate", "train", "eval"])
    ap.add_argument("--gate", type=int, default=0)
    ap.add_argument("--base_dir", default="/workspace/sft/base")
    ap.add_argument("--corpus", default="/workspace/sft/data/arianna_dataset_final_clean.txt")
    ap.add_argument("--ckpt_dir", default="/workspace/sft/checkpoints")
    ap.add_argument("--log_dir", default="/workspace/sft/logs")
    ap.add_argument("--eval_dir", default="/workspace/sft/eval")
    ap.add_argument("--adapter", default=None, help="adapter .pt for --mode eval")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=128.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_seq", type=int, default=512)
    ap.add_argument("--val_every", type=int, default=50)
    ap.add_argument("--backup_cmd", default=None,
                    help="Shell template, {src} = ckpt path. e.g. 'scp {src} neo:~/arianna/...'")
    args = ap.parse_args()

    set_seed()

    if args.mode == "gate":
        gates = {1: gate_1_tokenizer, 2: gate_2_base_load, 3: gate_3_lora_inject,
                 4: gate_4_lora_grad_smoke, 5: gate_5_mask_correctness, 6: gate_6_split,
                 7: gate_7_generate, 8: gate_8_neo_cpu_minirun}
        if args.gate not in gates:
            print(f"unknown gate {args.gate}; valid: {list(gates.keys())}")
            sys.exit(1)
        gates[args.gate](args)
        return

    if args.mode == "train":
        cmd_train(args)
        return

    if args.mode == "eval":
        cmd_eval(args)
        return


if __name__ == "__main__":
    main()
