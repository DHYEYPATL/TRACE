/* AssetWatch AI — dashboard JS */
const $ = (id) => document.getElementById(id);
let appState = null;
let selected  = null;
let lastEventId = 0;
let notificationsEnabled = false;
let pointerStart = null;
let dragRect     = null;
const canvas = $('selection-canvas');
const ctx    = canvas.getContext('2d');

// ── Intro modal ────────────────────────────────────────────────────
(function initIntro() {
  if (localStorage.getItem('aw_intro_seen')) return;
  $('intro-modal').classList.remove('hidden');
})();
$('intro-close').onclick = closeIntro;
$('intro-ok').onclick    = closeIntro;
function closeIntro() {
  if ($('intro-dont-show').checked) localStorage.setItem('aw_intro_seen', '1');
  $('intro-modal').classList.add('hidden');
}
$('show-help').onclick = () => $('intro-modal').classList.remove('hidden');

// ── Utilities ──────────────────────────────────────────────────────
function toast(message, type='') {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = message;
  $('toast-region').appendChild(el);
  setTimeout(() => el.remove(), 4300);
}
function prettyTime(value) {
  if (!value) return '—';
  const d = new Date(value);
  return Number.isNaN(d.valueOf()) ? value : d.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'});
}
function statusClass(s) {
  return ({PRESENT:'status-present',MOVED:'status-moved',CHECKING:'status-checking',
           MISSING:'status-missing',UNVERIFIED:'status-unverified'})[s] || 'status-unknown';
}
function escapeHtml(v) {
  return String(v ?? '').replace(/[&<>"']/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
async function api(url, body, method) {
  const m = method || (body === undefined ? 'GET' : 'POST');
  const r = await fetch(url, {
    method: m,
    headers: {'Content-Type':'application/json'},
    body: body === undefined ? undefined : JSON.stringify(body)
  });
  const data = await r.json();
  if (!r.ok || data.ok === false) throw new Error(data.error || `Request failed (${r.status})`);
  return data;
}

// ── Coverage meter ─────────────────────────────────────────────────
function coverageMeterHTML(gallerySize) {
  const max   = 10;
  const pct   = Math.min(100, Math.round((gallerySize / max) * 100));
  const color = pct >= 60 ? '#169b69' : pct >= 30 ? '#d38b21' : '#c94646';
  const hint  = gallerySize < 2
    ? ' · Add reference views for better tracking'
    : gallerySize < 5
    ? ' · Add more angles for viewpoint robustness'
    : gallerySize < 8
    ? ' · Good coverage'
    : ' · Excellent coverage';
  const tip = 'Aim for 4+ different angles (front, ±45°, side) for best reliability when the object moves or rotates.';
  return `<div class="coverage-wrap" title="${tip}">
    <span class="coverage-label">Gallery ${gallerySize}/${max}${hint}</span>
    <div class="coverage-bar"><div class="coverage-fill" style="width:${pct}%;background:${color}"></div></div>
  </div>`;
}

// ── Selection ──────────────────────────────────────────────────────
function showSelection(sel) {
  selected = sel;
  $('register-btn').disabled = !sel;
  if (!sel) {
    $('selection-info').classList.remove('active');
    $('selection-info').textContent = 'No object selected. Click a box or draw around an object.';
    return;
  }
  $('selection-info').classList.add('active');
  const mode = sel.mode === 'detector'
    ? `AI detection · ${sel.class_name}`
    : 'Manual visual tracking · arbitrary object';
  $('selection-info').textContent = `Selected: ${mode}. Enter a name below, then register.`;
  if (!$('asset-category').value && sel.class_name && sel.class_name !== 'custom')
    $('asset-category').value = sel.class_name;
}
function drawSelection(rect) {
  const img    = $('video-feed');
  const width  = appState?.runtime?.dimensions?.width  || img.naturalWidth  || 1280;
  const height = appState?.runtime?.dimensions?.height || img.naturalHeight || 720;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width; canvas.height = height;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (rect) {
    ctx.fillStyle   = 'rgba(73,119,247,.14)';
    ctx.fillRect(rect.x1, rect.y1, rect.x2-rect.x1, rect.y2-rect.y1);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth   = Math.max(2, width / 650);
    ctx.setLineDash([9, 5]);
    ctx.strokeRect(rect.x1, rect.y1, rect.x2-rect.x1, rect.y2-rect.y1);
    ctx.setLineDash([]);
  }
}

// ── Event alerts ───────────────────────────────────────────────────
function eventAlert(event) {
  if (!event || !event.event_id || event.event_id <= lastEventId) return;
  if (lastEventId !== 0 && ['MISSING','MOVED','RECOVERED','PROMOTED'].includes(event.event_type)) {
    if (notificationsEnabled && 'Notification' in window && Notification.permission === 'granted')
      new Notification('AssetWatch AI', {body: event.message});
    const type = event.event_type === 'MISSING' ? 'error'
               : event.event_type === 'RECOVERED' ? 'success' : '';
    toast(event.message, type);
    try {
      const audio = new AudioContext();
      const osc   = audio.createOscillator();
      const gain  = audio.createGain();
      osc.connect(gain); gain.connect(audio.destination);
      osc.frequency.value = event.event_type === 'MISSING' ? 520
                          : event.event_type === 'RECOVERED' ? 880 : 760;
      gain.gain.setValueAtTime(.0001, audio.currentTime);
      gain.gain.exponentialRampToValueAtTime(.07,   audio.currentTime + .02);
      gain.gain.exponentialRampToValueAtTime(.0001, audio.currentTime + .22);
      osc.start(); osc.stop(audio.currentTime + .24);
    } catch (_) {}
  }
  lastEventId = Math.max(lastEventId, event.event_id);
}

// ── Render assets ──────────────────────────────────────────────────
function renderAssets(assets) {
  $('asset-count').textContent   = assets.length;
  const counts = assets.reduce((o, a) => (o[a.status] = (o[a.status]||0)+1, o), {});
  $('present-count').textContent =
    `${counts.PRESENT||0} present · ${counts.MISSING||0} missing · ${counts.MOVED||0} moved`;
  $('status-counter').textContent = assets.length;
  if (!assets.length) {
    $('assets-list').innerHTML = '<div class="empty-state">Register your first item from the live feed.</div>';
    return;
  }
  $('assets-list').innerHTML = assets.map(a => {
    const recovBadge = a.recovered_from_gallery
      ? '<span class="recovered-badge">↩ Recovered</span>' : '';
    const confStr = a.identity_confidence != null
      ? `<span title="Identity confidence">ID ${Number(a.identity_confidence).toFixed(2)}</span>` : '';
    const gs = a.gallery_size ?? 0;
    return `
    <div class="asset-row${a.recovered_from_gallery ? ' asset-recovered' : ''}">
      <div class="asset-top">
        <div>
          <div class="asset-name">${escapeHtml(a.name)}${recovBadge}</div>
          <div class="asset-id">${escapeHtml(a.asset_id)} · ${escapeHtml(a.category)}</div>
        </div>
        <span class="status-pill ${statusClass(a.status)}">${escapeHtml(a.status)}</span>
      </div>
      ${coverageMeterHTML(gs)}
      <div class="asset-meta">
        <span>${a.tracker_mode === 'csrt' ? 'Manual tracker' : 'AI + appearance'}</span>
        <span>Last seen ${prettyTime(a.last_seen)}</span>
        ${a.last_similarity != null ? `<span>Match ${Number(a.last_similarity).toFixed(2)}</span>` : ''}
        ${confStr}
      </div>
      <div class="asset-actions">
        <button class="btn btn-secondary btn-small" data-ref="${escapeHtml(a.asset_id)}">＋ Reference view</button>
        <button class="btn btn-quiet btn-small"     data-archive="${escapeHtml(a.asset_id)}">Archive</button>
        <button class="btn btn-danger btn-small"    data-delete="${escapeHtml(a.asset_id)}" title="Permanently delete all data">🗑</button>
      </div>
    </div>`;
  }).join('');

  document.querySelectorAll('[data-ref]').forEach(btn => btn.onclick = async () => {
    try {
      if (!selected) throw new Error('Select another view of this item in the camera first.');
      await api(`/api/reference/${btn.dataset.ref}`, {});
      showSelection(null);
      toast('Reference view added — gallery updated.', 'success');
      await refresh();
    } catch(e) { toast(e.message, 'error'); }
  });
  document.querySelectorAll('[data-archive]').forEach(btn => btn.onclick = async () => {
    if (!confirm('Archive this item? History is retained but monitoring stops.')) return;
    try { await api(`/api/archive/${btn.dataset.archive}`, {}); toast('Item archived.'); await refresh(); }
    catch(e) { toast(e.message, 'error'); }
  });
  document.querySelectorAll('[data-delete]').forEach(btn => btn.onclick = async () => {
    if (!confirm('PERMANENTLY DELETE this item? All data, events and crop images will be removed. This cannot be undone.')) return;
    try { await api(`/api/delete/${btn.dataset.delete}`, {}); toast('Item permanently deleted.'); await refresh(); }
    catch(e) { toast(e.message, 'error'); }
  });
}

// ── Render events ──────────────────────────────────────────────────
function renderEvents(events) {
  $('event-counter').textContent = events.length;
  if (!events.length) {
    $('events-list').innerHTML = '<div class="empty-state">No events yet. Movement and missing-item alerts appear here.</div>';
    return;
  }
  $('events-list').innerHTML = events.map(e => `
    <div class="event-row">
      <div class="event-top">
        <span class="event-type">${escapeHtml(e.event_type)}</span>
        <span class="event-time">${prettyTime(e.created_at)}</span>
      </div>
      <div class="event-message">${escapeHtml(e.message)}</div>
    </div>`).join('');
}

// ── Render state ───────────────────────────────────────────────────
function renderState(data) {
  appState = data;
  const r  = data.runtime;
  if (r.dimensions?.width && r.dimensions?.height)
    $('video-wrap').style.aspectRatio = `${r.dimensions.width} / ${r.dimensions.height}`;
  $('capture-fps').textContent  = r.camera_healthy ? r.capture_fps : '—';
  $('infer-fps').textContent    = r.model_ready    ? r.infer_fps   : '—';
  $('latency').textContent      = r.model_ready    ? `${r.infer_latency_ms} ms` : 'waiting';
  $('device-name').textContent  = r.device || 'Detecting';
  $('device-foot').textContent  = r.gpu_available
    ? 'CUDA acceleration enabled'
    : (r.cuda_detected ? 'CUDA failed · CPU fallback' : 'CPU fallback · CUDA unavailable');
  if (document.activeElement !== $('camera-index'))
    $('camera-index').value = String(r.camera_index);
  $('camera-dot').className    = `status-dot ${r.camera_healthy ? 'good' : 'bad'}`;
  $('camera-status').textContent = r.camera_healthy
    ? `Camera ${r.camera_index} connected`
    : (r.camera_error || 'Waiting for camera');

  // Monitoring mode — always sync from server (even if user changed it elsewhere).
  const mon = r.monitoring || {};
  $('mode-select').value = mon.mode || 'always';
  if (mon.idle) $('idle-badge').classList.remove('hidden');
  else          $('idle-badge').classList.add('hidden');

  const warn = $('runtime-warning');
  if (r.model_error) {
    warn.textContent = `AI issue: ${r.model_error}. Camera streaming OK.`;
    warn.classList.remove('hidden');
  } else if (!r.camera_healthy && r.camera_error) {
    warn.textContent = `${r.camera_error} Alerts suspended.`;
    warn.classList.remove('hidden');
  } else {
    warn.classList.add('hidden');
  }

  renderAssets(data.assets);
  renderEvents(data.events);
  if (data.events.length) eventAlert(data.events[0]);
  drawSelection(dragRect);
}

async function refresh() {
  try { renderState(await api('/api/state')); }
  catch (e) { $('camera-status').textContent = 'Dashboard disconnected'; }
}

// ── Pointer events (selection canvas) ─────────────────────────────
$('video-wrap').addEventListener('pointerdown', (e) => {
  if (e.target !== canvas) return;
  const rect = canvas.getBoundingClientRect();
  const sx = canvas.width / rect.width, sy = canvas.height / rect.height;
  const x  = (e.clientX - rect.left) * sx, y = (e.clientY - rect.top) * sy;
  pointerStart = {x, y};
  dragRect = {x1:x, y1:y, x2:x, y2:y};
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener('pointermove', (e) => {
  if (!pointerStart) return;
  const rect = canvas.getBoundingClientRect();
  const sx = canvas.width/rect.width, sy = canvas.height/rect.height;
  const x = (e.clientX-rect.left)*sx, y = (e.clientY-rect.top)*sy;
  dragRect = {x1:Math.min(pointerStart.x,x),y1:Math.min(pointerStart.y,y),
              x2:Math.max(pointerStart.x,x),y2:Math.max(pointerStart.y,y)};
  drawSelection(dragRect);
});
canvas.addEventListener('pointerup', async (e) => {
  if (!pointerStart) return;
  const start = pointerStart; pointerStart = null;
  const rect  = canvas.getBoundingClientRect();
  const sx    = canvas.width/rect.width, sy = canvas.height/rect.height;
  const x     = (e.clientX-rect.left)*sx, y = (e.clientY-rect.top)*sy;
  const w     = Math.abs(x-start.x), h = Math.abs(y-start.y);
  try {
    const payload = w > 12 && h > 12
      ? {x1:Math.min(x,start.x),y1:Math.min(y,start.y),x2:Math.max(x,start.x),y2:Math.max(y,start.y)}
      : {x:Math.round(x),y:Math.round(y)};
    const data = await api('/api/select', payload);
    showSelection(data.selection); drawSelection(null); dragRect = null;
  } catch(e) { showSelection(null); drawSelection(null); dragRect=null; toast(e.message,'error'); }
});

// ── Buttons ────────────────────────────────────────────────────────
$('register-btn').onclick = async () => {
  const name = $('asset-name').value.trim();
  if (!name) { toast('Enter a name for the selected item.', 'error'); return; }
  try {
    const data = await api('/api/register', {name, category: $('asset-category').value.trim()});
    $('asset-name').value = ''; $('asset-category').value = '';
    showSelection(null);
    toast(`${data.asset.name} registered. Add reference views for better viewpoint coverage.`, 'success');
    await refresh();
  } catch(e) { toast(e.message, 'error'); }
};

$('switch-camera').onclick = async () => {
  try {
    await api('/api/camera', {index: Number($('camera-index').value)});
    toast('Reconnecting to selected camera…');
  } catch(e) { toast(e.message, 'error'); }
};

$('mode-select').onchange = async () => {
  try {
    await api('/api/mode', {mode: $('mode-select').value});
    toast(`Monitoring mode: ${$('mode-select').value}`);
  } catch(e) { toast(e.message, 'error'); }
};

$('enable-alerts').onclick = async () => {
  if (!('Notification' in window)) {
    toast('Browser notifications not supported. In-dashboard alerts still work.');
    return;
  }
  const result = await Notification.requestPermission();
  notificationsEnabled = result === 'granted';
  $('enable-alerts').textContent = notificationsEnabled ? 'Alerts enabled' : 'Alerts blocked';
  toast(notificationsEnabled ? 'Desktop alerts enabled.' : 'Permission not granted.', notificationsEnabled?'success':'');
};

// ── Poll ───────────────────────────────────────────────────────────
refresh();
setInterval(refresh, 900);
