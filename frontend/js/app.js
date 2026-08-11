// lyricsLM front end — calls the server-side model instead of running
// inference in the browser. Replace API_BASE with your deployed
// Vercel backend URL once you have it.
const API_BASE = 'https://lyrics-lm.vercel.app';

const chatLog = document.getElementById('chat-log');
const seedInput = document.getElementById('seed-input');
const sendBtn = document.getElementById('send-btn');
const tempSlider = document.getElementById('temp-slider');
const tempReadout = document.getElementById('temp-readout');
const statusLine = document.getElementById('status-line');

function moodLabel(t) {
  if (t <= 0.5) return 'focused';
  if (t <= 0.9) return 'balanced';
  if (t <= 1.2) return 'loose';
  return 'wild';
}

function updateTempReadout() {
  const t = parseFloat(tempSlider.value);
  tempReadout.innerHTML = `${t.toFixed(1)} <span class="temp-mood">· ${moodLabel(t)}</span>`;
}
tempSlider.addEventListener('input', updateTempReadout);
updateTempReadout();

function addBubble(role, text) {
  const bubble = document.createElement('div');
  bubble.className = `bubble ${role}`;
  const inner = document.createElement('div');
  inner.className = 'bubble-inner';
  inner.textContent = text;
  bubble.appendChild(inner);
  chatLog.appendChild(bubble);
  chatLog.scrollTop = chatLog.scrollHeight;
  return inner;
}

async function handleGenerate() {
  const seed = seedInput.value.trim();
  if (!seed) return;

  addBubble('user', seed);
  seedInput.value = '';
  sendBtn.disabled = true;
  statusLine.textContent = 'Generating… (first request after a while may take ~30s to wake the server)';

  const modelBubble = addBubble('model', '…');

  try {
    const res = await fetch(`${API_BASE}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        seed,
        max_new_tokens: 200,
        temperature: parseFloat(tempSlider.value),
      }),
    });

    if (!res.ok) {
      throw new Error(`Server returned ${res.status}`);
    }

    const data = await res.json();
    modelBubble.textContent = data.text || '(empty response)';
    statusLine.textContent = 'Ready.';
  } catch (err) {
    modelBubble.textContent = `Something went wrong: ${err.message}`;
    statusLine.textContent = 'Error — try again.';
  } finally {
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener('click', handleGenerate);
seedInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') handleGenerate();
});
