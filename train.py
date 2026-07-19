#!/usr/bin/env python
"""The crowd-trained tiny LLM — generation 5.

Auto-generated from the canonical slots at deep-ml.com/research/tiny-llm.
Trains from scratch on any UTF-8 text file and reports bits per byte on a
held-out tail. This is a faithful standalone version of the platform harness
(same slots, same wiring); the platform adds sandboxing + hidden eval data.

    python train.py --data your_text.txt [--steps 2000]
"""

import argparse, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace


def _rh_validate_merges(merges, vocab_cap):
    if not isinstance(merges, list):
        raise ValueError("merges must be a list of (id, id) pairs")
    if 256 + len(merges) > int(vocab_cap):
        raise ValueError(f"vocab too large: 256+{len(merges)} exceeds cap {vocab_cap}")
    clean = []
    for i, pr in enumerate(merges):
        if not (isinstance(pr, (list, tuple)) and len(pr) == 2):
            raise ValueError(f"merge {i} is not a pair")
        a, b = pr
        if not (isinstance(a, int) and isinstance(b, int)):
            raise ValueError(f"merge {i} ids must be ints")
        hi = 256 + i
        if not (0 <= a < hi and 0 <= b < hi):
            raise ValueError(f"merge {i} references id >= {hi}")
        clean.append((a, b))
    return clean


def _rh_encode(byts, merges):
    """Apply merges in order, greedy left-to-right per merge. Trusted."""
    arr = np.frombuffer(byts, dtype=np.uint8).astype(np.int32)
    for i, (a, b) in enumerate(merges):
        if len(arr) < 2:
            break
        m = (arr[:-1] == a) & (arr[1:] == b)
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            continue
        keep = []
        last = -2
        for j in idx:
            if j > last + 1:
                keep.append(j)
                last = j
        keep = np.asarray(keep)
        arr[keep] = 256 + i
        arr = np.delete(arr, keep + 1)
    return arr.astype(np.int64)


def _rh_token_byte_lens(merges):
    lens = [1] * 256
    for a, b in merges:
        lens.append(lens[a] + lens[b])
    return np.asarray(lens, dtype=np.int64)


# --- slot: config (v5, by Giuseppe Frigeni) ---
def configure_model(cfg):
    """The model's shape and training hyperparameters (vanilla nanoGPT).

    Everything here is a tradeoff against the fixed compute budget: deeper
    or wider means fewer steps before the wall clock; longer context means
    slower steps but more to attend to. The param cap is the hard ceiling.
    """
    cfg.n_layer = 6
    cfg.n_head = 6
    cfg.n_embd = 384
    cfg.block_size = 96      # context length (max 1024)
    cfg.dropout = 0.1
    cfg.batch_size = 96
    cfg.learning_rate = 1e-3
    return cfg

# --- slot: tokenizer (v1, by moe chabot) ---
def build_tokenizer(train_bytes):
    """Classic BPE: repeatedly merge the most frequent adjacent pair.

    Learns on a 1MB sample so it stays fast inside the wall clock; the
    trusted encoder applies the merges to the full corpus afterwards.
    512 merges -> vocab 768, well under the 2048 cap.
    """
    NUM_MERGES = 512
    arr = np.frombuffer(train_bytes[:1_000_000], dtype=np.uint8).astype(np.int32)

    merges = []
    for i in range(NUM_MERGES):
        # Count adjacent pairs (encode each pair as one int; 4096 > vocab cap).
        pairs = arr[:-1].astype(np.int64) * 4096 + arr[1:].astype(np.int64)
        uniq, counts = np.unique(pairs, return_counts=True)
        top = counts.argmax()
        if counts[top] < 2:
            break
        a, b = int(uniq[top] // 4096), int(uniq[top] % 4096)
        merges.append((a, b))

        # Apply the merge greedily left-to-right, same as the trusted encoder.
        idx = np.flatnonzero((arr[:-1] == a) & (arr[1:] == b))
        keep, last = [], -2
        for j in idx:
            if j > last + 1:
                keep.append(j)
                last = j
        keep = np.asarray(keep, dtype=np.int64)
        arr[keep] = 256 + i
        arr = np.delete(arr, keep + 1)
    return merges

# --- slot: embeddings (v4, by Shubh Goyal) ---
class Embeddings(nn.Module):
    """Learned token + positional embeddings (vanilla nanoGPT)."""

    def __init__(self, cfg):
        super().__init__()
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        # Position ids live in a buffer, NOT torch.arange(..., device=...) in
        # forward: tracing would bake the device as a constant and the frozen
        # artifact could then only ever run on the device it was traced on.
        # Buffers are remapped by map_location, so the model stays portable.
        self.register_buffer("pos_ids", torch.arange(cfg.block_size))

    def forward(self, idx):
        return self.drop(self.tok(idx))

# --- slot: attention (v4, by Shubh Goyal) ---
class Attention(nn.Module):
    """Multi-head causal self-attention with Rotary Positional Embeddings (RoPE)."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_dropout = cfg.dropout

        # Precompute RoPE frequencies as a buffer (portable across devices,
        # same reasoning as pos_ids: never build device-pinned tensors in
        # forward()).
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(cfg.block_size).float()
        freqs = torch.outer(t, inv_freq)  # (block_size, head_dim/2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def _apply_rope(self, x, T):
        # x: (B, n_head, T, head_dim)
        cos = self.cos[:T].unsqueeze(0).unsqueeze(0)  # (1,1,T,head_dim/2)
        sin = self.sin[:T].unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        rot_x1 = x1 * cos - x2 * sin
        rot_x2 = x1 * sin + x2 * cos
        out = torch.stack([rot_x1, rot_x2], dim=-1).flatten(-2)
        return out

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = self._apply_rope(q, T)
        k = self._apply_rope(k, T)

        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

# --- slot: ffn (v2, by Nick Grebe) ---
class FFN(nn.Module):
    """Parameter-efficient SwiGLU feed-forward network."""

    def __init__(self, cfg):
        super().__init__()
        #Approximately matches the parameter count of a standard 4x FFN:
        hidden_dim = int((8 / 3) * cfg.n_embd)

        # Hardware-friendly rounding.
        hidden_dim = 64 * ((hidden_dim + 63) // 64)

        self.gate = nn.Linear(
            cfg.n_embd,
            hidden_dim,
            bias=False,
        )
        self.up = nn.Linear(
            cfg.n_embd,
            hidden_dim,
            bias=False,
        )
        self.down = nn.Linear(
            hidden_dim,
            cfg.n_embd,
            bias=False,
        )

        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = F.silu(self.gate(x)) * self.up(x)
        return self.drop(self.down(x))

# --- slot: norm (v4, by Shubh Goyal) ---
class Norm(nn.Module):
    """RMSNorm — cheaper than LayerNorm (no mean-subtraction, no bias),
    and typically gives slightly better loss at this scale."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight

# --- slot: architecture (v0, by Deep-ML) ---
class Block(nn.Module):
    """Pre-norm transformer block (vanilla nanoGPT wiring)."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = Norm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.ln2 = Norm(cfg.n_embd)
        self.ffn = FFN(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class GPT(nn.Module):
    """Embeddings -> N blocks -> final norm -> untied lm_head."""

    def __init__(self, cfg):
        super().__init__()
        self.embeddings = Embeddings(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = Norm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, idx):
        x = self.embeddings(idx)
        for b in self.blocks:
            x = b(x)
        return self.lm_head(self.norm_f(x))


def build_model(cfg):
    """The topology is yours: rewire blocks, tie weights, go parallel."""
    return GPT(cfg)

# --- slot: optimizer (v5, by Giuseppe Frigeni) ---
def configure_optimizer(model, cfg):
    """AdamW with weight decay on matrices only (vanilla nanoGPT)."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.03},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
    )

# --- slot: lr_schedule (v0, by Deep-ML) ---
def get_lr(step, cfg):
    """Linear warmup then cosine decay to 10% of base lr."""
    warmup = 100
    if step < warmup:
        return cfg.learning_rate * (step + 1) / warmup
    progress = (step - warmup) / max(1, cfg.max_steps - warmup)
    return cfg.learning_rate * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))

# --- slot: train_step (v0, by Deep-ML) ---
def train_step(model, batch, optimizer, step):
    """Plain cross-entropy step with grad clipping (vanilla nanoGPT)."""
    x, y = batch
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to a UTF-8 text file")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(1337)
    np.random.seed(1337)

    raw = open(args.data, "rb").read()
    split = int(len(raw) * 0.95)
    train_bytes, eval_bytes = raw[:split], raw[split:]

    merges = _rh_validate_merges(build_tokenizer(train_bytes), 2048)
    train = torch.from_numpy(_rh_encode(train_bytes, merges))
    lens = _rh_token_byte_lens(merges)

    cfg = SimpleNamespace(**{'n_layer': 6, 'batch_size': 64, 'n_embd': 384, 'dropout': 0.1, 'learning_rate': 0.0003, 'block_size': 256, 'n_head': 6})
    cfg = configure_model(cfg) or cfg
    cfg.vocab_size = 256 + len(merges)
    cfg.max_steps = args.steps
    cfg.device = args.device

    model = build_model(cfg).to(cfg.device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}  vocab: {cfg.vocab_size}")
    opt = configure_optimizer(model, cfg)

    gen = torch.Generator().manual_seed(1337)

    def get_batch():
        ix = torch.randint(len(train) - cfg.block_size - 1, (cfg.batch_size,), generator=gen)
        x = torch.stack([train[i : i + cfg.block_size] for i in ix])
        y = torch.stack([train[i + 1 : i + cfg.block_size + 1] for i in ix])
        return x.to(cfg.device), y.to(cfg.device)

    model.train()
    t0 = time.monotonic()
    for step in range(cfg.max_steps):
        lr = float(get_lr(step, cfg))
        for g in opt.param_groups:
            g["lr"] = lr
        loss = train_step(model, get_batch(), opt, step)
        if step % 50 == 0:
            print(f"step {step}  train_loss {loss:.4f}  lr {lr:.2e}  {time.monotonic()-t0:.0f}s")

    # bits per byte on the held-out tail
    model.eval()
    toks = torch.from_numpy(_rh_encode(eval_bytes, merges))
    T = cfg.block_size
    n = (len(toks) - 1) // T
    total_nll, total_bytes = 0.0, 0
    with torch.no_grad():
        for i in range(n):
            x = toks[i * T : i * T + T].unsqueeze(0).to(cfg.device)
            y = toks[i * T + 1 : i * T + T + 1].unsqueeze(0).to(cfg.device)
            logits = model(x)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
            total_nll += float(nll)
            total_bytes += int(lens[y.cpu().numpy()].sum())
    print(f"held-out bits per byte: {total_nll / math.log(2) / max(1, total_bytes):.4f}")


if __name__ == "__main__":
    main()
