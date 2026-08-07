# lyricsLM front end

A static, no-backend chatbot. It loads three model checkpoints as raw binary
files and runs the transformer forward pass in plain JavaScript, in a Web
Worker, entirely in the visitor's browser. Nothing is sent to a server.

Right now it ships with **placeholder weights** (randomly initialised, not
trained — the yellow banner on the page says so) so you can see and test the
whole UI before you've trained anything. Swap in your real checkpoints and
the banner disappears automatically.

## 1. Train and export in Colab

1. Open `lyricsLM.ipynb` (the version with the new **"12. Export the weights
   for the browser"** section added at the end) in Google Colab.
2. Runtime → Change runtime type → T4 GPU.
3. Runtime → Run all. Training takes about 10 minutes.
4. The new last cell writes `web_weights/` and zips it to
   `lyricsLM_web_weights.zip`. Download that zip from the Colab file browser
   (folder icon, left sidebar → find the file → ⋮ → Download).

That zip contains:

```
web_weights/
  config.json
  baby.bin              baby.manifest.json
  halfway.bin            halfway.manifest.json
  trained.bin            trained.manifest.json
```

## 2. Install the real weights into this front end

Unzip it, then for each checkpoint copy its `.bin`, its `.manifest.json`,
and `config.json` into the matching folder here:

```
weights/baby/      ← baby.bin, baby.manifest.json, config.json
weights/halfway/   ← halfway.bin, halfway.manifest.json, config.json
weights/trained/   ← trained.bin, trained.manifest.json, config.json
```

(That's the exact structure the placeholder files already use — just
overwrite them.) `config.json` is identical across all three; copy the same
one into each folder.

Reload the page. The demo banner disappears once it sees a real
`config.json` (the export script only sets `demo_placeholder: true` on the
placeholder set generated for this scaffold — your real export won't have
that flag at all, so nothing extra to do).

## 3. Deploy

This is a fully static site — `index.html`, `css/`, `js/`, `weights/`. Any
static host works:

- **Cloudflare Pages / Vercel (static)** — drag the whole folder in, or
  connect the repo. No build command needed.
- **GitHub Pages** — push this folder to a repo, enable Pages on the
  `main` branch.
- **Your own server** — copy the folder anywhere Nginx/Apache can serve
  static files.

Point `lyricsLM.brainbeam.ai` at wherever you deploy it. No backend, no
environment variables, no API keys.

**To test locally**, don't just double-click `index.html` — browsers block
`fetch()` of local files under the `file://` origin. Run a tiny local
server from this folder instead, then open the printed URL:

```
python3 -m http.server 8000
# then visit http://localhost:8000
```

## Notes on the engine

- `js/engine.js` runs in a Web Worker and implements: token + position
  embedding, six transformer blocks (multi-head causal self-attention,
  feed-forward, residual connections, LayerNorm), a final LayerNorm, and the
  output projection — the same architecture the notebook trains, read
  straight out of the exported tensors.
- It uses a **KV cache**: each new character only requires one incremental
  step through the stack instead of recomputing the whole sequence, so a
  180-character reply generates in under a second even on modest hardware.
  This is mathematically identical to the notebook's own `generate()`
  function — just the standard efficiency trick every production LLM uses,
  rather than a shortcut that changes the output.
- The model's context window is 128 characters (`block_size` in the
  notebook). Seed + generated text is capped at that; very long seed lines
  will leave less room for generation.
- Tensor names expected in each manifest follow PyTorch's `state_dict()`
  naming for the `LyricsLM` class in the notebook exactly
  (`token_embedding.weight`, `blocks.{i}.sa.heads.{h}.key.weight`, etc.).
  If you change the model architecture in the notebook, `engine.js` will
  need matching changes.
