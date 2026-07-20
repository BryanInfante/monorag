/* Inspector END — Chat client */

const thread = document.getElementById('chat-thread');
const scroll = document.getElementById('chat-scroll');
const emptyState = document.getElementById('empty-state');
const input = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');

const AVATAR_SVG = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.4" stroke-linecap="round"><circle cx="10.5" cy="10.5" r="6.5"></circle><line x1="15.5" y1="15.5" x2="21" y2="21"></line></svg>';
const SOURCE_ICON = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#4F46E5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><path d="M14 2v6h6"></path></svg>';

let isWaiting = false;

// --- Load status on startup ---
async function loadStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    document.getElementById('status-label').textContent = 'CONECTADO';
    document.getElementById('doc-count').textContent = data.document_count;
    document.getElementById('collection-name').textContent = data.collection;
    document.getElementById('model-name').textContent = data.model;
    document.getElementById('provider-name').textContent = data.provider;

    // Populate suggested questions
    const pillsContainer = document.getElementById('suggested-pills');
    const emptyPills = document.getElementById('empty-pills');
    data.suggested_questions.forEach(function(q) {
      const btn = document.createElement('button');
      btn.className = 'ie-pill';
      btn.textContent = q;
      btn.addEventListener('click', function() { submitQuestion(q); });
      pillsContainer.appendChild(btn);

      const btn2 = document.createElement('button');
      btn2.className = 'ie-empty-pill';
      btn2.textContent = q;
      btn2.addEventListener('click', function() { submitQuestion(q); });
      emptyPills.appendChild(btn2);
    });
  } catch (e) {
    document.getElementById('status-label').textContent = 'ERROR';
    document.getElementById('status-label').style.color = '#dc2626';
  }
}

// --- Submit question ---
async function submitQuestion(query) {
  if (!query.trim() || isWaiting) return;

  // Show chat, hide empty state
  emptyState.style.display = 'none';
  scroll.style.display = 'flex';

  // Add user bubble
  const userRow = document.createElement('div');
  userRow.className = 'ie-row-user';
  userRow.innerHTML = '<div class="ie-bubble-user">' + escapeHtml(query) + '</div>';
  thread.appendChild(userRow);

  // Add thinking indicator
  const thinking = document.createElement('div');
  thinking.className = 'ie-thinking';
  thinking.innerHTML = '<div class="ie-avatar">' + AVATAR_SVG + '</div><div class="ie-thinking-bubble"><span class="ie-spinner"></span><span class="ie-thinking-text">Pensando…</span></div>';
  thread.appendChild(thinking);
  scrollToBottom();

  isWaiting = true;
  sendBtn.disabled = true;
  input.value = '';

  try {
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query })
    });
    const data = await res.json();

    // Remove thinking
    thinking.remove();

    if (data.error) {
      appendError(data.error);
    } else {
      appendAssistantMessage(data.answer, data.sources || []);
    }
  } catch (e) {
    thinking.remove();
    appendError('Error de conexión. Intenta de nuevo en un momento.');
  }

  isWaiting = false;
  sendBtn.disabled = false;
  scrollToBottom();
}

// --- Render assistant message ---
function appendAssistantMessage(answer, sources) {
  const assistant = document.createElement('div');
  assistant.className = 'ie-assistant';

  let html = '<div class="ie-assistant-msg"><div class="ie-avatar">' + AVATAR_SVG + '</div><div class="ie-bubble-assistant">' + formatAnswer(answer) + '</div></div>';

  if (sources.length > 0) {
    html += '<div class="ie-sources">';
    html += '<button class="ie-sources-toggle" aria-expanded="false"><span class="caret">▸</span>' + SOURCE_ICON + ' Fuentes (' + sources.length + ')</button>';
    html += '<div class="ie-sources-list" hidden>';
    sources.forEach(function(src) {
      const meta = src.metadata || {};
      const file = meta.source || 'Unknown';
      const page = meta.page;
      const snippet = (src.text || '').substring(0, 200);
      html += '<div class="ie-source"><div class="ie-source-head"><span class="ie-source-file">' + escapeHtml(file) + '</span>';
      if (page && page !== 'N/A' && page !== 0) {
        html += '<span class="ie-source-page">p. ' + page + '</span>';
      }
      html += '</div><blockquote>' + escapeHtml(snippet) + '</blockquote></div>';
    });
    html += '</div></div>';
  }

  assistant.innerHTML = html;
  thread.appendChild(assistant);

  // Bind source toggle
  const toggle = assistant.querySelector('.ie-sources-toggle');
  if (toggle) {
    toggle.addEventListener('click', function() {
      const list = toggle.parentElement.querySelector('.ie-sources-list');
      const open = list.hidden;
      list.hidden = !open;
      toggle.setAttribute('aria-expanded', String(open));
      toggle.querySelector('.caret').textContent = open ? '▾' : '▸';
    });
  }
}

// --- Render error ---
function appendError(message) {
  const err = document.createElement('div');
  err.className = 'ie-error';
  err.textContent = message;
  thread.appendChild(err);
}

// --- Helpers ---
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function formatAnswer(text) {
  // Basic markdown-like: **bold** and newlines
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');
}

function scrollToBottom() {
  scroll.scrollTop = scroll.scrollHeight;
}

// --- Event listeners ---
sendBtn.addEventListener('click', function() {
  submitQuestion(input.value);
});

input.addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    submitQuestion(input.value);
  }
});

// --- Init ---
loadStatus();
