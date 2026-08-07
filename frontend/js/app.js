const WEIGHTS_BASE = 'weights';
const CHECKPOINTS = [
  { id: 'baby', label: 'Baby', sub: 'step 0' },
  { id: 'halfway', label: 'Halfway', sub: 'step 1,500' },
  { id: 'trained', label: 'Trained', sub: 'step 5,000' },
];

const worker = new Worker('js/engine.js');
const state = {
  checkpoint: 'trained',
  loaded: {},
  config: null,
  busy: false,
  currentBubble: null,
};

const el = {
  tabs: document.getElementById('checkpoint-tabs'),
  log: document.getElementById('chat-log'),
  form: document.getElementById('composer'),
  input: document.getElementById('seed-input'),
  temp: document.getElementById('temp-slider'),
  tempVal: document.getElementById('temp-value'),
  status: document.getElementById('status-line'),
  banner: document.getElementById('demo-banner'),
  sendBtn: document.getElementById('send-btn'),
  progressBar: document.getElementById('load-progress'),
  progressFill: document.getElementById('load-progress-fill'),
};

function renderTabs() {
  el.tabs.innerHTML = '';
  for (const cp of CHECKPOINTS) {
    const btn = document.createElement('button');
    const isLoading = state.loadingId === cp.id;
    btn.className = 'tab' + (cp.id === state.checkpoint ? ' active' : '') + (isLoading ? ' loading' : '');
    btn.innerHTML = `<span class="tab-label">${cp.label}</span><span class="tab-sub">${cp.sub}</span>`;
    btn.addEventListener('click', () => switchCheckpoint(cp.id));
    el.tabs.appendChild(btn);
  }
}

function switchCheckpoint(id) {
  if (state.busy) return;
  state.checkpoint = id;
  renderTabs();
  ensureLoaded(id);
}

function ensureLoaded(id) {
  if (state.loaded[id]) {
    setStatus(`${labelFor(id)} ready \u00b7 ${fmtParams(state.loaded[id].paramCount)} parameters`);
    return;
  }
  state.loadingId = id;
  setBusy(true);
  renderTabs();
  el.progressBar.hidden = false;
  el.progressFill.style.width = '0%';
  setStatus(`Loading ${labelFor(id)} checkpoint\u2026`);
  worker.postMessage({ type: 'load', base: `${WEIGHTS_BASE}/${id}`, name: id });
}

function labelFor(id) {
  return CHECKPOINTS.find(c => c.id === id)?.label ?? id;
}

function fmtParams(n) {
  return (n / 1e6).toFixed(2) + 'M';
}

function setStatus(text) {
  el.status.textContent = text;
}

function addBubble(role, text) {
  const wrap = document.createElement('div');
  wrap.className = `bubble ${role}`;
  const inner = document.createElement('div');
  inner.className = 'bubble-inner';
  inner.textContent = text;
  wrap.appendChild(inner);
  el.log.appendChild(wrap);
  el.log.scrollTop = el.log.scrollHeight;
  return inner;
}

worker.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === 'progress') {
    const pct = Math.min(100, Math.round((msg.loaded / msg.total) * 100));
    el.progressFill.style.width = pct + '%';
    if (msg.name === state.checkpoint) {
      setStatus(`Loading ${labelFor(msg.name)} checkpoint\u2026 ${pct}% \u00b7 ${(msg.loaded / 1e6).toFixed(1)}MB / ${(msg.total / 1e6).toFixed(1)}MB`);
    }
  } else if (msg.type === 'loaded') {
    state.config = msg.config;
    state.loaded[msg.name] = { paramCount: msg.paramCount };
    if (msg.config.demo_placeholder) {
      el.banner.hidden = false;
    }
    state.loadingId = null;
    el.progressBar.hidden = true;
    renderTabs();
    if (msg.name === state.checkpoint) {
      setStatus(`${labelFor(msg.name)} ready \u00b7 ${fmtParams(msg.paramCount)} parameters`);
    }
    setBusy(false);
  } else if (msg.type === 'token') {
    if (state.currentBubble) {
      state.currentBubble.textContent += msg.char;
      el.log.scrollTop = el.log.scrollHeight;
    }
  } else if (msg.type === 'done') {
    setBusy(false);
    setStatus(`${labelFor(state.checkpoint)} ready \u00b7 ${fmtParams(state.loaded[state.checkpoint].paramCount)} parameters`);
  } else if (msg.type === 'error') {
    setBusy(false);
    state.loadingId = null;
    el.progressBar.hidden = true;
    renderTabs();
    if (state.currentBubble) state.currentBubble.textContent = '(' + msg.message + ')';
    setStatus('Something went sideways \u2014 try a different line.');
  }
};

function setBusy(b) {
  state.busy = b;
  el.sendBtn.disabled = b;
  el.input.disabled = b;
  document.querySelectorAll('.tab').forEach(t => t.disabled = b);
}

el.temp.addEventListener('input', () => {
  const v = parseFloat(el.temp.value).toFixed(2);
  el.tempVal.textContent = v;
  el.tempVal.dataset.mood =
    v < 0.6 ? 'careful' : v > 1.1 ? 'wild' : 'neutral';
  el.tempVal.nextElementSibling && (el.tempVal.nextElementSibling.textContent = el.tempVal.dataset.mood);
});

el.form.addEventListener('submit', (e) => {
  e.preventDefault();
  if (state.busy) return;
  const seed = el.input.value.trim();
  if (!seed) return;
  if (!state.loaded[state.checkpoint]) {
    setStatus('Still loading the model \u2014 one second\u2026');
    return;
  }
  addBubble('user', seed);
  state.currentBubble = addBubble('model', '');
  setBusy(true);
  setStatus(`${labelFor(state.checkpoint)} is writing\u2026`);
  worker.postMessage({
    type: 'generate',
    seed,
    maxNewTokens: 180,
    temperature: parseFloat(el.temp.value),
  });
  el.input.value = '';
});

// init
renderTabs();
setBusy(true);
ensureLoaded(state.checkpoint);
