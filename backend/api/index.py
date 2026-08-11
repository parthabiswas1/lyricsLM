"""
lyricsLM backend — pure NumPy inference, no PyTorch.

PyTorch (even CPU-only) exceeds Vercel's 500MB function size limit
because of bundled MKL libraries. Since this model is tiny (2.7M
params) and we only need a forward pass (no training/autograd),
NumPy alone is sufficient and keeps the deployed bundle under 50MB.

Weights are exported from the trained PyTorch checkpoint into a
plain .npz file (see backend_export_cell.py) — this file loads that
directly, with no PyTorch import anywhere.
"""
import os
import json
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

with open(os.path.join(MODEL_DIR, "config.json")) as f:
    CFG = json.load(f)

vocab_size = CFG["vocab_size"]
n_embd = CFG["n_embd"]
n_head = CFG["n_head"]
n_layer = CFG["n_layer"]
block_size = CFG["block_size"]
stoi = CFG["stoi"]
itos = {int(k): v for k, v in CFG["itos"].items()}
head_size = n_embd // n_head

W = np.load(os.path.join(MODEL_DIR, "trained_weights.npz"))


# ---------------------------------------------------------------------
# NumPy building blocks — mirror the PyTorch model's forward pass
# ---------------------------------------------------------------------
def layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def linear(x, weight, bias=None):
    # PyTorch nn.Linear stores weight as (out_features, in_features)
    out = x @ weight.T
    if bias is not None:
        out = out + bias
    return out


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def relu(x):
    return np.maximum(x, 0)


def attention_head(x, block_idx, head_idx, T):
    prefix = f"blocks.{block_idx}.sa.heads.{head_idx}"
    k = linear(x, W[f"{prefix}.key.weight"])
    q = linear(x, W[f"{prefix}.query.weight"])
    v = linear(x, W[f"{prefix}.value.weight"])

    wei = q @ k.transpose(0, 2, 1) * (k.shape[-1] ** -0.5)
    mask = np.tril(np.ones((T, T)))
    wei = np.where(mask == 0, -1e10, wei)
    wei = softmax(wei, axis=-1)
    return wei @ v


def multi_head_attention(x, block_idx, T):
    heads_out = [attention_head(x, block_idx, h, T) for h in range(n_head)]
    out = np.concatenate(heads_out, axis=-1)
    prefix = f"blocks.{block_idx}.sa"
    return linear(out, W[f"{prefix}.proj.weight"], W[f"{prefix}.proj.bias"])


def feed_forward(x, block_idx):
    prefix = f"blocks.{block_idx}.ff.net"
    x = linear(x, W[f"{prefix}.0.weight"], W[f"{prefix}.0.bias"])
    x = relu(x)
    x = linear(x, W[f"{prefix}.2.weight"], W[f"{prefix}.2.bias"])
    return x


def block_forward(x, block_idx, T):
    ln1_w, ln1_b = W[f"blocks.{block_idx}.ln1.weight"], W[f"blocks.{block_idx}.ln1.bias"]
    ln2_w, ln2_b = W[f"blocks.{block_idx}.ln2.weight"], W[f"blocks.{block_idx}.ln2.bias"]
    x = x + multi_head_attention(layer_norm(x, ln1_w, ln1_b), block_idx, T)
    x = x + feed_forward(layer_norm(x, ln2_w, ln2_b), block_idx)
    return x


def model_forward(idx):
    B, T = idx.shape
    tok_emb = W["token_embedding.weight"][idx]
    pos_emb = W["position_embedding.weight"][:T]
    x = tok_emb + pos_emb
    for i in range(n_layer):
        x = block_forward(x, i, T)
    x = layer_norm(x, W["ln_f.weight"], W["ln_f.bias"])
    logits = linear(x, W["head.weight"], W["head.bias"])
    return logits


def encode(s):
    return [stoi[c] for c in s if c in stoi]


def decode(nums):
    return "".join(itos[n] for n in nums)


def generate(seed, max_new_tokens=800, temperature=0.8, min_lines=4):
    seed_ids = encode(seed)
    if not seed_ids:
        return ""
    idx = np.array([seed_ids], dtype=np.int64)
    lines_seen = 0
    current_line_chars = []
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model_forward(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-4)
        probs = softmax(logits, axis=-1)[0]
        nxt = int(np.random.choice(len(probs), p=probs))
        idx = np.concatenate([idx, np.array([[nxt]])], axis=1)
        ch = itos.get(nxt, "")
        if ch == "\n":
            # Only count lines that actually have text — blank spacer lines
            # between stanzas shouldn't count toward the minimum.
            if "".join(current_line_chars).strip():
                lines_seen += 1
            current_line_chars = []
            if lines_seen >= min_lines:
                break
        else:
            current_line_chars.append(ch)
    return decode(idx[0].tolist())


# ---------------------------------------------------------------------
# API
# ---------------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://lyricslm.brainbeam.ai",
        "https://lyricslm.pages.dev",
        "http://localhost:8000",
    ],
    allow_methods=["POST"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    seed: str
    max_new_tokens: int = 200
    temperature: float = 0.8


@app.post("/api/generate")
def generate_endpoint(req: GenerateRequest):
    max_tokens = min(max(req.max_new_tokens, 1), 800)
    temperature = min(max(req.temperature, 0.1), 2.0)
    text = generate(req.seed, max_tokens, temperature)
    return {"text": text}


@app.get("/api/generate")
def health():
    total_params = sum(arr.size for arr in W.values())
    return {"status": "ok", "vocab_size": vocab_size, "params": int(total_params)}
