"""
lyricsLM backend — Vercel Python serverless function.

Reconstructs the exact PyTorch model architecture from the training
notebook and serves POST /api/generate. Loads the trained checkpoint
once per cold start (module-level), so warm requests are fast.
"""
import os
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------
# Load config (vocab, architecture hyperparameters) exported from Colab
# ---------------------------------------------------------------------
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
dropout = 0.0  # inference only, no dropout

device = "cpu"


# ---------------------------------------------------------------------
# Model — identical architecture to the training notebook
# ---------------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        w = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)
        return w @ self.value(x)


class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHead(n_head, n_embd // n_head)
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class LyricsLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(T, device=idx.device))
        x = self.blocks(tok + pos)
        x = self.ln_f(x)
        return self.head(x)


# ---------------------------------------------------------------------
# Load trained weights once per cold start
# ---------------------------------------------------------------------
model = LyricsLM().to(device)
state = torch.load(os.path.join(MODEL_DIR, "trained.pt"), map_location="cpu")
model.load_state_dict(state)
model.eval()


def encode(s):
    return [stoi[c] for c in s if c in stoi]


def decode(nums):
    return "".join(itos[n] for n in nums)


@torch.no_grad()
def generate(seed, max_new_tokens=200, temperature=0.8):
    seed_ids = encode(seed)
    if not seed_ids:
        return ""
    idx = torch.tensor([seed_ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-4)
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, nxt), dim=1)
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
        "http://localhost:8000",  # for local testing
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
    max_tokens = min(max(req.max_new_tokens, 1), 300)  # clamp, avoid abuse
    temperature = min(max(req.temperature, 0.1), 2.0)
    text = generate(req.seed, max_tokens, temperature)
    return {"text": text}


@app.get("/api/generate")
def health():
    return {"status": "ok", "vocab_size": vocab_size, "params": sum(p.numel() for p in model.parameters())}
