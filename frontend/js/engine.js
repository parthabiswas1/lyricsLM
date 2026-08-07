// lyricsLM inference engine — runs entirely in this Web Worker.
// No PyTorch, no ONNX, no server: just the same arithmetic the article
// walks through by hand, written out in plain JavaScript over typed arrays.

let CFG = null;
let TENSORS = null; // name -> Float32Array view
let LOADED_NAME = null;

// ---- tensor loading -------------------------------------------------

async function fetchWithProgress(url, name) {
  const res = await fetch(url);
  const total = Number(res.headers.get('Content-Length')) || 0;
  if (!res.body || !total) {
    // fall back silently if streaming or content-length isn't available
    return await res.arrayBuffer();
  }
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    self.postMessage({ type: 'progress', name, loaded: received, total });
  }
  const full = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    full.set(chunk, offset);
    offset += chunk.length;
  }
  return full.buffer;
}

async function loadCheckpoint(base, name) {
  const [manifest, bin, config] = await Promise.all([
    fetch(`${base}/${name}.manifest.json`).then(r => r.json()),
    fetchWithProgress(`${base}/${name}.bin`, name),
    CFG ? Promise.resolve(CFG) : fetch(`${base}/config.json`).then(r => r.json()),
  ]);
  CFG = config;
  const full = new Float32Array(bin);
  const tensors = {};
  for (const entry of manifest) {
    tensors[entry.name] = full.subarray(entry.offset, entry.offset + entry.length);
  }
  TENSORS = tensors;
  LOADED_NAME = name;
  return { config: CFG, paramCount: full.length };
}

function T(name) {
  const t = TENSORS[name];
  if (!t) throw new Error(`missing tensor: ${name}`);
  return t;
}

// ---- math helpers -----------------------------------------------------

// y = W x + b   where W is [outDim, inDim] row-major (PyTorch nn.Linear layout)
function linear(x, W, b, outDim, inDim, out) {
  out = out || new Float32Array(outDim);
  for (let o = 0; o < outDim; o++) {
    let sum = b ? b[o] : 0;
    const rowOff = o * inDim;
    for (let i = 0; i < inDim; i++) sum += W[rowOff + i] * x[i];
    out[o] = sum;
  }
  return out;
}

function layerNorm(x, weight, bias, n) {
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let variance = 0;
  for (let i = 0; i < n; i++) {
    const d = x[i] - mean;
    variance += d * d;
  }
  variance /= n;
  const inv = 1 / Math.sqrt(variance + 1e-5);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * weight[i] + bias[i];
  return out;
}

function softmaxInPlace(arr) {
  let max = -Infinity;
  for (const v of arr) if (v > max) max = v;
  let sum = 0;
  for (let i = 0; i < arr.length; i++) {
    arr[i] = Math.exp(arr[i] - max);
    sum += arr[i];
  }
  for (let i = 0; i < arr.length; i++) arr[i] /= sum;
  return arr;
}

// ---- model state (KV cache) -------------------------------------------
// One entry per layer, growing as tokens are generated.
// cache[layer].k[head] = array of Float32Array(headSize), one per position.

let cache = null;
let positions = 0;

function resetCache() {
  const { n_layer, n_head } = CFG;
  cache = [];
  for (let l = 0; l < n_layer; l++) {
    const heads = [];
    for (let h = 0; h < n_head; h++) heads.push({ k: [], v: [] });
    cache.push(heads);
  }
  positions = 0;
}

// Run one token through the whole stack, updating the cache, and return logits.
function forwardStep(tokenId) {
  const { n_embd, n_head, n_layer, block_size, vocab_size } = CFG;
  const headSize = n_embd / n_head;
  const pos = positions;
  if (pos >= block_size) throw new Error('context window full');

  const tokEmb = T('token_embedding.weight').subarray(tokenId * n_embd, tokenId * n_embd + n_embd);
  const posEmb = T('position_embedding.weight').subarray(pos * n_embd, pos * n_embd + n_embd);
  let x = new Float32Array(n_embd);
  for (let i = 0; i < n_embd; i++) x[i] = tokEmb[i] + posEmb[i];

  for (let l = 0; l < n_layer; l++) {
    const p = `blocks.${l}`;
    // --- self attention ---
    const xn = layerNorm(x, T(`${p}.ln1.weight`), T(`${p}.ln1.bias`), n_embd);
    const attnOut = new Float32Array(n_embd);
    for (let h = 0; h < n_head; h++) {
      const kW = T(`${p}.sa.heads.${h}.key.weight`);
      const qW = T(`${p}.sa.heads.${h}.query.weight`);
      const vW = T(`${p}.sa.heads.${h}.value.weight`);
      const k = linear(xn, kW, null, headSize, n_embd);
      const q = linear(xn, qW, null, headSize, n_embd);
      const v = linear(xn, vW, null, headSize, n_embd);

      const hc = cache[l][h];
      hc.k.push(k);
      hc.v.push(v);

      const T_now = hc.k.length; // positions 0..pos, inclusive
      const scores = new Float32Array(T_now);
      const scale = 1 / Math.sqrt(headSize);
      for (let t = 0; t < T_now; t++) {
        let dot = 0;
        const kt = hc.k[t];
        for (let d = 0; d < headSize; d++) dot += q[d] * kt[d];
        scores[t] = dot * scale;
      }
      softmaxInPlace(scores);

      const outHead = new Float32Array(headSize);
      for (let t = 0; t < T_now; t++) {
        const w = scores[t];
        const vt = hc.v[t];
        for (let d = 0; d < headSize; d++) outHead[d] += w * vt[d];
      }
      attnOut.set(outHead, h * headSize);
    }
    const proj = linear(attnOut, T(`${p}.sa.proj.weight`), T(`${p}.sa.proj.bias`), n_embd, n_embd);
    for (let i = 0; i < n_embd; i++) x[i] += proj[i];

    // --- feed forward ---
    const xn2 = layerNorm(x, T(`${p}.ln2.weight`), T(`${p}.ln2.bias`), n_embd);
    const hidden = linear(xn2, T(`${p}.ff.net.0.weight`), T(`${p}.ff.net.0.bias`), 4 * n_embd, n_embd);
    for (let i = 0; i < hidden.length; i++) if (hidden[i] < 0) hidden[i] = 0; // ReLU
    const ffOut = linear(hidden, T(`${p}.ff.net.2.weight`), T(`${p}.ff.net.2.bias`), n_embd, 4 * n_embd);
    for (let i = 0; i < n_embd; i++) x[i] += ffOut[i];
  }

  const xf = layerNorm(x, T('ln_f.weight'), T('ln_f.bias'), n_embd);
  const logits = linear(xf, T('head.weight'), T('head.bias'), vocab_size, n_embd);

  positions++;
  return logits;
}

function sample(logits, temperature) {
  const scaled = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) scaled[i] = logits[i] / Math.max(temperature, 1e-4);
  softmaxInPlace(scaled);
  let r = Math.random();
  let acc = 0;
  for (let i = 0; i < scaled.length; i++) {
    acc += scaled[i];
    if (r <= acc) return i;
  }
  return scaled.length - 1;
}

// ---- worker message protocol ------------------------------------------

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'load') {
      const info = await loadCheckpoint(msg.base, msg.name);
      self.postMessage({ type: 'loaded', name: msg.name, config: info.config, paramCount: info.paramCount });
      return;
    }
    if (msg.type === 'generate') {
      const { seed, maxNewTokens, temperature } = msg;
      resetCache();
      const { stoi, itos, block_size } = CFG;
      const seedChars = Array.from(seed).filter(c => c in stoi);
      if (seedChars.length === 0) {
        self.postMessage({ type: 'error', message: 'None of those characters are in this model\u2019s vocabulary yet.' });
        return;
      }
      let lastLogits = null;
      for (const ch of seedChars) {
        lastLogits = forwardStep(stoi[ch]);
      }
      const budget = Math.max(0, Math.min(maxNewTokens, block_size - seedChars.length));
      let out = '';
      for (let n = 0; n < budget; n++) {
        const nextId = sample(lastLogits, temperature);
        const ch = itos[String(nextId)];
        out += ch;
        self.postMessage({ type: 'token', char: ch });
        lastLogits = forwardStep(nextId);
      }
      self.postMessage({ type: 'done', text: out });
      return;
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: err.message });
  }
};
