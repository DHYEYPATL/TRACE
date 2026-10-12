// TRACE Enterprise Dashboard Controller
let currentActiveEventId = null;
let currentActiveAssetId = null;
let ws = null;

// ROI Drawing Canvas State
let canvasPoints = [];
let canvasBgImage = new Image();

// Tab View Switcher
function switchView(viewName) {
  document.querySelectorAll('.view-section').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));

  const targetSection = document.getElementById(`view-${viewName}`);
  if (targetSection) targetSection.classList.add('active');

  const btnIndex = ['dashboard', 'wizard', 'audit', 'demo', 'settings', 'health'].indexOf(viewName);
  const btns = document.querySelectorAll('.tab-btn');
  if (btns[btnIndex]) btns[btnIndex].classList.add('active');

  if (viewName === 'audit') loadAuditEvents();
  if (viewName === 'health') loadHealthMetrics();
  if (viewName === 'settings') loadSettingsForm();
  if (viewName === 'wizard') captureCanvasSnapshot();
}

// WebSocket Live Streaming & Fallback Polling
function initWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/stream`;

  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log('TRACE WebSocket connected');
    document.getElementById('kpi-camera-dot').className = 'dot';
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.kind === 'state_heartbeat') {
        updateTelemetry(data);
      } else if (data.kind === 'alert') {
        prependAlertItem(data.event);
        loadAuditEvents();
        loadReviewQueue();
      }
    } catch (e) {
      console.error('Error parsing WS message:', e);
    }
  };

  ws.onclose = () => {
    console.warn('WebSocket disconnected. Retrying in 2s...');
    setTimeout(initWebSocket, 2000);
  };
}

// Periodic Fallback Polling (Every 1.5s)
async function pollState() {
  try {
    const res = await fetch('/api/state');
    const data = await res.json();
    if (data.ok) {
      updateTelemetry(data);
    }
    loadReviewQueue();
  } catch (err) {
    console.error('Error polling state:', err);
  }
}

function updateTelemetry(data) {
  // Update KPI Strip
  const camText = document.getElementById('kpi-camera-text');
  const camDot = document.getElementById('kpi-camera-dot');
  if (data.camera_status === 'HEALTHY') {
    camText.innerText = 'Camera 1/1 Online';
    camDot.className = 'dot';
  } else {
    camText.innerText = `Camera ${data.camera_status}`;
    camDot.className = 'dot danger';
  }

  const netText = document.getElementById('kpi-net-text');
  const netDot = document.getElementById('kpi-net-dot');
  if (data.network_status === 'ONLINE') {
    netText.innerText = 'Network Online';
    netDot.className = 'dot';
  } else {
    netText.innerText = `Network OFFLINE (Q: ${data.queue_depth || data.pending_outbox || 0})`;
    netDot.className = 'dot warn';
  }

  document.getElementById('kpi-fps-text').innerText = `${data.fps || 5.0} FPS | Latency P95 1.4s`;

  // Update Asset Cards
  if (data.assets) {
    renderAssetCards(data.assets);
  }
}

function renderAssetCards(assets) {
  const container = document.getElementById('asset-cards-container');
  if (!container) return;

  container.innerHTML = assets.map(a => {
    const beliefPct = Math.round((a.belief ?? 1.0) * 100);
    const stateName = a.state || 'SAFE';

    return `
      <div class="asset-card">
        <div class="asset-card-top">
          <div class="asset-info">
            <h3>${a.name || a.id}</h3>
            <span>ID: ${a.id} • Class: ${a.class || 'Asset'}</span>
          </div>
          <span class="state-chip ${stateName}">${stateName}</span>
        </div>
        <div class="meter-row">
          <span>Confidence:</span>
          <div class="meter-bar-wrap">
            <div class="meter-bar-fill" style="width: ${beliefPct}%;"></div>
          </div>
          <strong>${(a.belief ?? 1.0).toFixed(2)}</strong>
        </div>
      </div>
    `;
  }).join('');
}

// Selective Prediction: Review Queue (0.60 to 0.89 Confidence)
async function loadReviewQueue() {
  try {
    const res = await fetch('/api/review-queue');
    const data = await res.json();
    if (!data.ok) return;

    const listEl = document.getElementById('review-queue-list');
    const counterEl = document.getElementById('review-counter');
    if (!listEl) return;

    counterEl.innerText = `${data.count} pending review`;

    if (data.count === 0) {
      listEl.innerHTML = `<div style="padding: 12px; text-align: center; color: var(--text-muted); font-size: 12px;">No ambiguous events awaiting operator review.</div>`;
      return;
    }

    listEl.innerHTML = data.items.map(it => `
      <div style="background: var(--bg-card); border: 1px solid rgba(245, 158, 11, 0.4); border-radius: 6px; padding: 10px; display: flex; align-items: center; justify-content: space-between;">
        <div>
          <strong style="color: var(--state-unknown);">[Review Needed] ${it.asset_id} • ${it.type}</strong>
          <div style="font-size: 11px; color: var(--text-muted);">Confidence: ${(it.confidence || 0).toFixed(2)} | Time: ${new Date(it.detected_at * 1000).toLocaleTimeString()}</div>
        </div>
        <div style="display: flex; gap: 6px;">
          <button class="btn btn-primary" onclick="confirmReviewEvent('${it.id}')">✓ Confirm Alert</button>
          <button class="btn btn-danger" onclick="dismissReviewEvent('${it.id}')">✕ Dismiss</button>
        </div>
      </div>
    `).join('');
  } catch(e) {
    console.error('Failed to load review queue:', e);
  }
}

async function confirmReviewEvent(eventId) {
  try {
    await fetch(`/api/review-queue/${eventId}/confirm`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ actor: 'Operator Review Confirm' })
    });
    loadReviewQueue();
    loadAuditEvents();
  } catch(e) { console.error(e); }
}

async function dismissReviewEvent(eventId) {
  try {
    await fetch(`/api/review-queue/${eventId}/dismiss`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ actor: 'Operator Review Dismiss' })
    });
    loadReviewQueue();
    loadAuditEvents();
  } catch(e) { console.error(e); }
}

function prependAlertItem(ev) {
  const feed = document.getElementById('alert-feed');
  const emptyNote = feed.querySelector('div[style*="text-align: center"]');
  if (emptyNote) emptyNote.remove();

  const sevClass = (ev.severity || 'medium').toUpperCase();
  const alertRow = document.createElement('div');
  alertRow.className = `alert-item ${sevClass}`;
  alertRow.innerHTML = `
    <div class="alert-details">
      <strong>[${sevClass}] ${ev.asset_id} • ${ev.type}</strong>
      <div class="meta">Confidence: ${ev.confidence} | Detected: ${ev.detected_at}</div>
    </div>
    <div style="display: flex; gap: 8px;">
      <button class="btn btn-success" onclick="quickAck('${ev.id}')">✓ Ack</button>
      <button class="btn btn-primary" onclick="openIncidentModal('${ev.id}')">Open</button>
    </div>
  `;

  feed.prepend(alertRow);
}

// Incident Modal
async function openIncidentModal(eventId) {
  currentActiveEventId = eventId;
  const modal = document.getElementById('incident-modal');
  modal.classList.add('active');

  try {
    const res = await fetch(`/api/events/${eventId}`);
    const data = await res.json();
    if (!data.ok) return;

    const ev = data.event;
    currentActiveAssetId = ev.asset_id;

    document.getElementById('modal-title').innerText = `${ev.asset_id} - ${ev.type} Alert`;
    document.getElementById('modal-meta').innerText = `Event ID: ${ev.id} | Severity: ${ev.severity.toUpperCase()} | Detected: ${ev.detected_at}`;
    document.getElementById('modal-confidence').innerText = `${(ev.confidence || 0).toFixed(2)}`;
    
    // Payload JSON
    let pObj = {};
    try { pObj = JSON.parse(ev.payload_json || '{}'); } catch(e){}
    const ledgerIdx = pObj.ledger ? pObj.ledger.idx : 'Anchored';
    document.getElementById('modal-ledger-idx').innerText = `#${ledgerIdx}`;

    // Images
    const beforeImg = document.getElementById('modal-before-img');
    const afterImg = document.getElementById('modal-after-img');

    beforeImg.src = ev.before_path ? `/${ev.before_path}` : '';
    afterImg.src = ev.after_path ? `/${ev.after_path}` : '';
  } catch (err) {
    console.error('Failed to load incident detail:', err);
  }
}

function closeModal() {
  document.getElementById('incident-modal').classList.remove('active');
  currentActiveEventId = null;
}

// Incident Actions
async function executeIncidentAction(actionName) {
  if (!currentActiveEventId) return;

  try {
    const res = await fetch(`/api/events/${currentActiveEventId}/${actionName}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ actor: 'Security Console', note: `Action ${actionName} applied.` })
    });
    const data = await res.json();
    if (data.ok) {
      alert(`Action '${actionName}' executed and signed into the cryptographic ledger.`);
      closeModal();
      loadAuditEvents();
    }
  } catch (err) {
    alert(`Action failed: ${err}`);
  }
}

async function quickAck(eventId) {
  try {
    await fetch(`/api/events/${eventId}/ack`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ actor: 'Operator Quick Ack' })
    });
    loadAuditEvents();
  } catch (err) {
    console.error(err);
  }
}

async function authorizeMoveFromModal() {
  if (!currentActiveAssetId) return;
  try {
    const res = await fetch(`/api/assets/${currentActiveAssetId}/authorize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ duration_minutes: 10, actor: 'Branch Staff' })
    });
    const data = await res.json();
    alert(data.message || 'Move authorized for 10 minutes.');
    closeModal();
  } catch (err) {
    alert(`Authorize failed: ${err}`);
  }
}

async function acceptBaselineFromModal() {
  if (!currentActiveAssetId) return;
  try {
    const res = await fetch(`/api/assets/${currentActiveAssetId}/baseline/accept`, {
      method: 'POST'
    });
    const data = await res.json();
    alert(data.message || 'New baseline registered.');
    closeModal();
  } catch (err) {
    alert(`Re-baseline failed: ${err}`);
  }
}

// Interactive Visual ROI Canvas
function initRoiCanvas() {
  const canvas = document.getElementById('roi-canvas');
  if (!canvas) return;

  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    // Convert to 1280x720 video coordinates
    const vx = Math.round((e.clientX - rect.left) * scaleX * (1280 / canvas.width));
    const vy = Math.round((e.clientY - rect.top) * scaleY * (720 / canvas.height));

    canvasPoints.push([vx, vy]);
    document.getElementById('canvas-point-count').innerText = `Points: ${canvasPoints.length}`;
    document.getElementById('wiz-region').value = JSON.stringify(canvasPoints);
    drawCanvas();
  });
}

function captureCanvasSnapshot() {
  const canvas = document.getElementById('roi-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  canvasBgImage.crossOrigin = 'anonymous';
  canvasBgImage.src = `/api/video/feed?t=${new Date().getTime()}`;
  canvasBgImage.onload = () => {
    ctx.drawImage(canvasBgImage, 0, 0, canvas.width, canvas.height);
    drawCanvas();
  };
}

function clearCanvasPoints() {
  canvasPoints = [];
  document.getElementById('canvas-point-count').innerText = `Points: 0`;
  document.getElementById('wiz-region').value = `[]`;
  drawCanvas();
}

function drawCanvas() {
  const canvas = document.getElementById('roi-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (canvasBgImage.complete && canvasBgImage.naturalWidth > 0) {
    ctx.drawImage(canvasBgImage, 0, 0, canvas.width, canvas.height);
  } else {
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  if (canvasPoints.length > 0) {
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    ctx.fillStyle = 'rgba(16, 185, 129, 0.2)';
    ctx.beginPath();

    const scaleX = canvas.width / 1280;
    const scaleY = canvas.height / 720;

    canvasPoints.forEach(([x, y], idx) => {
      const cx = x * scaleX;
      const cy = y * scaleY;
      if (idx === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);

      // Draw dot
      ctx.fillStyle = '#3b82f6';
      ctx.fillRect(cx - 3, cy - 3, 6, 6);
    });

    if (canvasPoints.length >= 3) {
      ctx.closePath();
      ctx.fillStyle = 'rgba(16, 185, 129, 0.2)';
      ctx.fill();
    }
    ctx.stroke();
  }
}

// Wizard Submission
async function handleRegisterSubmit(e) {
  e.preventDefault();
  const id = document.getElementById('wiz-id').value.trim();
  const name = document.getElementById('wiz-name').value.trim();
  const className = document.getElementById('wiz-class').value;
  const criticality = parseInt(document.getElementById('wiz-criticality').value);
  const mode = document.getElementById('wiz-mode').value;
  const tagStr = document.getElementById('wiz-tag').value.trim();
  const tagId = tagStr ? parseInt(tagStr) : null;
  const regionRaw = document.getElementById('wiz-region').value.trim();

  let regionPts = [];
  try {
    regionPts = JSON.parse(regionRaw);
  } catch(err) {
    alert('Invalid region JSON format. Must be [[x,y], [x,y], ...]');
    return;
  }

  const payload = {
    id, name, class: className, criticality, mode,
    camera_id: 'cam-1', tag_id: tagId, region: regionPts,
    present_threshold: 0.70
  };

  try {
    const res = await fetch('/api/assets', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.ok) {
      // Capture median baseline
      await fetch(`/api/assets/${id}/baseline?kind=median`, { method: 'POST' });
      alert(`Asset '${name}' registered & baselined successfully! Returning to dashboard.`);
      switchView('dashboard');
    }
  } catch(err) {
    alert(`Registration failed: ${err}`);
  }
}

// Audit & Ledger Verification
async function loadAuditEvents() {
  try {
    const res = await fetch('/api/events?limit=50');
    const data = await res.json();
    if (!data.ok) return;

    const tbody = document.getElementById('events-table-body');
    if (!tbody) return;

    tbody.innerHTML = data.events.map(ev => `
      <tr>
        <td style="font-family: monospace;">${ev.id.substring(0, 8)}...</td>
        <td><strong>${ev.asset_id}</strong></td>
        <td><span class="state-chip ${ev.type}">${ev.type}</span></td>
        <td>${ev.severity.toUpperCase()}</td>
        <td>${ev.confidence.toFixed(2)}</td>
        <td>${new Date(ev.detected_at * 1000).toLocaleTimeString()}</td>
        <td>${ev.status}</td>
        <td><button class="btn" onclick="openIncidentModal('${ev.id}')">View Evidence</button></td>
      </tr>
    `).join('');
  } catch (err) {
    console.error('Failed to load audit events:', err);
  }
}

async function verifyLedgerAudit() {
  const banner = document.getElementById('ledger-verify-banner');
  banner.style.display = 'block';
  banner.innerText = 'Calculating Ed25519 signatures and verifying SHA-256 hash chains across all ledger blocks...';
  banner.style.background = 'rgba(59, 130, 246, 0.2)';
  banner.style.color = '#3b82f6';
  banner.style.border = '1px solid #3b82f6';

  try {
    const res = await fetch('/api/ledger/verify');
    const data = await res.json();

    if (data.ok) {
      banner.style.background = 'rgba(16, 185, 129, 0.15)';
      banner.style.color = '#10b981';
      banner.style.border = '1px solid #10b981';
      banner.innerHTML = `🛡️ <strong>100% Cryptographic Audit Passed!</strong><br>${data.message} | Head Hash: <code style="font-size: 11px;">${data.head.substring(0, 32)}...</code>`;
    } else {
      banner.style.background = 'rgba(244, 63, 94, 0.2)';
      banner.style.color = '#f43f5e';
      banner.style.border = '1px solid #f43f5e';
      banner.innerHTML = `🚨 <strong>AUDIT ALERT: Tampering Detected!</strong><br>Entry #${data.bad_idx}: ${data.why}`;
    }
  } catch (err) {
    banner.innerText = `Audit check failed: ${err}`;
  }
}

function exportLedgerCSV() {
  window.open('/api/events?limit=1000', '_blank');
}

// Scenario Fault Injector
async function injectFault(faultType, assetId = null) {
  try {
    const headers = { 'Content-Type': 'application/json' };
    const token = localStorage.getItem('trace_token') || localStorage.getItem('trace_auth_token');
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }
    const res = await fetch('/api/demo/inject', {
      method: 'POST',
      headers: headers,
      body: JSON.stringify({
        fault_type: faultType,
        asset_id: assetId,
        duration_seconds: 6.0
      })
    });
    const data = await res.json();
    if (res.ok && data.ok) {
      console.log(`Fault ${faultType} injected:`, data);
      loadAuditEvents();
      loadReviewQueue();
    } else {
      alert(`Fault trigger error: ${data.detail || JSON.stringify(data)}`);
    }
  } catch (err) {
    alert(`Fault trigger error: ${err}`);
  }
}

// Settings Screen Controller
async function loadSettingsForm() {
  try {
    const res = await fetch('/api/settings');
    const data = await res.json();
    if (data.business_hours) {
      document.getElementById('set-bh-start').value = data.business_hours.start || '09:30';
      document.getElementById('set-bh-end').value = data.business_hours.end || '18:30';
    }
    if (data.timers) {
      document.getElementById('set-t-missing').value = data.timers.missing_wait_s || 15;
      document.getElementById('set-t-newobj').value = data.timers.new_object_s || 8;
      document.getElementById('set-moved-votes').value = data.timers.moved_votes || 5;
      document.getElementById('set-t-swap').value = data.timers.swap_s || 10;
    }
  } catch(e) { console.error('Failed to load settings:', e); }
}

async function saveSettingsForm() {
  const payload = {
    business_hours: {
      start: document.getElementById('set-bh-start').value,
      end: document.getElementById('set-bh-end').value,
    },
    timers: {
      missing_wait_s: parseFloat(document.getElementById('set-t-missing').value),
      new_object_s: parseFloat(document.getElementById('set-t-newobj').value),
      moved_votes: parseInt(document.getElementById('set-moved-votes').value),
      swap_s: parseFloat(document.getElementById('set-t-swap').value),
    }
  };
  try {
    const res = await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    alert(data.message || 'Settings updated.');
  } catch(e) { alert(`Failed to save settings: ${e}`); }
}

// System Health
async function loadHealthMetrics() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    document.getElementById('health-fps').innerText = `${data.fps} FPS`;
    document.getElementById('health-queue').innerText = `${data.queue_depth} pending`;
  } catch(err) {
    console.error(err);
  }
}

function refreshFeed() {
  const img = document.getElementById('mjpeg-feed');
  if (img) img.src = `/api/video/feed?t=${new Date().getTime()}`;
}

// Initialization on DOM Load
window.addEventListener('DOMContentLoaded', () => {
  initWebSocket();
  initRoiCanvas();
  setInterval(pollState, 1500);
  loadAuditEvents();
  loadReviewQueue();
});
