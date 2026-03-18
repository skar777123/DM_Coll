/**
 * dashboard.js
 * ────────────
 * Real-time SocketIO client for BlindSpotGuard dashboard.
 *
 * Handles:
 *  - WebSocket connection / reconnection
 *  - System state updates → zone cards, indicators, vehicle diagram
 *  - Camera frame updates → live image elements
 *  - Event log with timestamped entries (colour-coded by severity)
 *  - Clock, uptime, stats counters
 *  - Alert banner for critical events
 *  - Manual override buttons
 */

'use strict';

/* ═══════════════════════ CONFIG ═══════════════════════════ */
const ZONE_SAFE     = 'safe';
const ZONE_CAUTION  = 'caution';
const ZONE_CRITICAL = 'critical';

const MAX_LOG_ENTRIES = 120;
const ALERT_DURATION  = 3500;   // ms

/* ═══════════════════════ STATE ════════════════════════════ */
let totalAlerts   = 0;
let updateCount   = 0;
let uptimeStart   = Date.now();
let lastUpdateMin = Date.now();
let updatesThisMin = 0;
let alertTimeout  = null;
let connected     = false;

/* ═══════════════════════ SOCKET ═══════════════════════════ */
const socket = io({ transports: ['websocket', 'polling'] });

socket.on('connect', () => {
  connected = true;
  setConnectionState('connected');
  log('Connected to BlindSpotGuard Pi.', 'info');
  socket.emit('request_frame', { position: 'left' });
  socket.emit('request_frame', { position: 'right' });
  socket.emit('request_frame', { position: 'rear' });
});

socket.on('disconnect', () => {
  connected = false;
  setConnectionState('connecting');
  log('Lost connection — reconnecting…', 'caution');
});

socket.on('state_update', (data) => {
  applyStateUpdate(data);
  updateCount++;
  updatesThisMin++;
  document.getElementById('statUpdates').textContent = updatesThisMin;
});

socket.on('camera_frame', (data) => {
  applyFrame(data);
});

socket.on('override', (data) => {
  log(`Manual override: ${data.direction} → ${data.zone}`, data.zone === 'safe' ? 'safe' : 'critical');
});

/* Measure latency every 5 s */
setInterval(() => {
  const t = Date.now();
  socket.emit('ping_latency', t, (ts) => {
    const ping = Date.now() - (ts || t);
    document.getElementById('statPing').textContent = ping;
  });
}, 5000);

/* Reset updates/min counter */
setInterval(() => {
  document.getElementById('statUpdates').textContent = updatesThisMin;
  updatesThisMin = 0;
}, 60000);

/* ═══════════════════════ STATE UPDATE ═════════════════════ */
function applyStateUpdate(data) {
  if (!data) return;

  const dirs = ['left', 'right', 'rear'];
  let anyCritical = false;
  let criticalDirs = [];

  dirs.forEach(dir => {
    const d = data[dir];
    if (!d) return;

    const prevZone = getZone(dir);
    setZone(dir, d.zone, d.distance_cm, d.camera_threat, d.led_mode, d.motor_mode, d.is_vehicle, d.is_moving, d.vision_active);

    if (d.zone === ZONE_CRITICAL && prevZone !== ZONE_CRITICAL) {
      anyCritical = true;
      criticalDirs.push(dir.toUpperCase());
      totalAlerts++;
      document.getElementById('statAlerts').textContent = totalAlerts;
      log(`⚠ CRITICAL: ${dir.toUpperCase()} | ${d.distance_cm.toFixed(1)} cm`, 'critical');
    } else if (d.zone === ZONE_CAUTION && prevZone === ZONE_SAFE) {
      log(`⚡ CAUTION: ${dir.toUpperCase()} | ${d.distance_cm.toFixed(1)} cm`, 'caution');
    } else if (d.zone === ZONE_SAFE && prevZone !== ZONE_SAFE) {
      log(`✔ ${dir.toUpperCase()} cleared — Safe`, 'safe');
    }
  });

  if (anyCritical) {
    showAlert(`⚠ CRITICAL THREAT — ${criticalDirs.join(' & ')}`);
  }

  // Rear motor indicators
  const rearMotorActive = data.rear && data.rear.motor_mode === 'pulse';
  setIndicatorDot('motor-rear-l', rearMotorActive ? 'active-motor' : '');
  setIndicatorDot('motor-rear-r', rearMotorActive ? 'active-motor' : '');
}

/* Zone elements cache */
const _zoneCache = {};
function zoneEl(id) {
  if (!_zoneCache[id]) _zoneCache[id] = document.getElementById(id);
  return _zoneCache[id];
}

function getZone(dir) {
  const card = zoneEl('card-' + dir);
  if (!card) return 'safe';
  for (const cls of card.classList) {
    if (cls.startsWith('zone-')) return cls.replace('zone-', '');
  }
  return 'safe';
}

function setZone(dir, zone, distCm, camThreat, ledMode, motorMode, isVehicle, isMoving, visionActive) {
  const card    = zoneEl('card-' + dir);
  const distEl  = zoneEl('dist-' + dir);
  const barEl   = zoneEl('bar-' + dir);
  const badgeEl = zoneEl('badge-' + dir);
  const arcEl   = zoneEl('arc-' + dir);
  if (!card) return;

  // Card class
  card.className = card.className.replace(/zone-\w+/g, '').trim();
  card.classList.add('zone-' + zone);
  // Ensure base classes are present
  if (!card.classList.contains('monitor-card')) card.classList.add('monitor-card');
  if (dir === 'rear' && !card.classList.contains('monitor-card--rear')) card.classList.add('monitor-card--rear');

  // Distance
  if (distEl) {
    if (zone === 'offline') {
      distEl.textContent = 'OFFLINE';
    } else {
      distEl.textContent = distCm >= 400 ? '> 400 cm' : `${distCm.toFixed(1)} cm`;
    }
  }

  // Progress bar
  if (barEl) {
    const pct = Math.max(2, Math.min(100, ((400 - distCm) / 400) * 100));
    barEl.style.width = pct + '%';
  }

  // Badge
  if (badgeEl) {
    badgeEl.textContent = zone.toUpperCase();
  }

  // Vehicle arc
  if (arcEl) {
    arcEl.className = `threat-arc arc-${dir} zone-${zone}`;
  }

  // LED indicator
  const ledDot = zoneEl('led-' + dir);
  if (ledDot) {
    ledDot.className = 'ind-dot led-dot' +
      (ledMode === 'flash' ? ' flash-led' : ledMode === 'solid' ? ' active-led' : '');
  }

  // Motor indicator (left/right only — rear handled separately)
  if (dir !== 'rear') {
    setIndicatorDot('motor-' + dir, motorMode === 'pulse' ? 'active-motor' : '');
  }

  // Camera vision status indicator
  let camClass = '';
  if (camThreat) {
    camClass = 'active-cam';
  } else if (visionActive) {
    camClass = 'vision-ok';
  } else {
    camClass = 'vision-off';
  }
  setIndicatorDot('cam-threat-' + dir, camClass);
  
  // Vehicle indicator
  setIndicatorDot('is-vehicle-' + dir, isVehicle ? 'active-vehicle' : '');
  
  // Moving indicator
  setIndicatorDot('is-moving-' + dir, isMoving ? 'active-moving' : '');
}

function setIndicatorDot(id, extraClass) {
  const el = zoneEl(id);
  if (!el) return;
  el.className = 'ind-dot' + (extraClass ? ' ' + extraClass : '');
  // Preserve original type class
  if (id.startsWith('led-'))           el.classList.add('led-dot');
  else if (id.startsWith('motor-'))    el.classList.add('motor-dot');
  else if (id.startsWith('cam-'))      el.classList.add('cam-dot');
  else if (id.startsWith('is-vehicle')) el.classList.add('vehicle-dot');
  else if (id.startsWith('is-moving'))  el.classList.add('moving-dot');
}

/* ═══════════════════════ CAMERA FRAMES ════════════════════ */
const _camElements = {
  left:  { img: null, overlay: null, det: null, card: null, status: null },
  right: { img: null, overlay: null, det: null, card: null, status: null },
  rear:  { img: null, overlay: null, det: null, card: null, status: null },
};
function initCamRefs() {
  ['left','right','rear'].forEach(pos => {
    const p = pos.charAt(0).toUpperCase() + pos.slice(1);
    _camElements[pos].img     = document.getElementById('cam' + p);
    _camElements[pos].overlay = document.getElementById('camOverlay' + p);
    _camElements[pos].det     = document.getElementById('camDet' + p);
    _camElements[pos].card    = document.getElementById('card-' + pos);
  });
}

function applyFrame(data) {
  if (!data || !data.position) return;
  const pos = data.position;
  const refs = _camElements[pos];
  if (!refs || !refs.img) return;

  // Update image
  if (data.frame_b64) {
    refs.img.src = 'data:image/jpeg;base64,' + data.frame_b64;
    if (refs.overlay) refs.overlay.classList.add('hidden');
  }

  // Threat border/glow
  if (refs.card) {
    refs.card.classList.toggle('threat-active', !!data.threat);
  }

  // Detections
  if (refs.det && data.detections) {
    if (data.detections.length === 0) {
      refs.det.innerHTML = '<span style="color:var(--text-mute)">No objects</span>';
    } else {
      refs.det.innerHTML = data.detections
        .map(d => {
          return `<span class="det-chip">${d.label} ${Math.round(d.confidence * 100)}%</span>`;
        })
        .join('');
    }
  }
}

/* ═══════════════════════ EVENT LOG ════════════════════════ */
const LOG_TYPES = { info: 'log-info', caution: 'log-caution', critical: 'log-critical', safe: 'log-safe' };

function log(message, type = 'info') {
  const container = document.getElementById('eventLog');
  if (!container) return;
  const now      = new Date();
  const ts       = now.toTimeString().slice(0, 8);
  const entry    = document.createElement('div');
  entry.className = 'log-entry ' + (LOG_TYPES[type] || 'log-info');
  entry.textContent = `[${ts}] ${message}`;
  container.appendChild(entry);
  // Auto-scroll
  container.scrollTop = container.scrollHeight;
  // Trim old entries
  while (container.children.length > MAX_LOG_ENTRIES) {
    container.removeChild(container.firstChild);
  }
}

document.getElementById('btnClearLog').addEventListener('click', () => {
  const container = document.getElementById('eventLog');
  container.innerHTML = '';
  log('Log cleared.', 'info');
});

/* ═══════════════════════ ALERT BANNER ═════════════════════ */
function showAlert(message) {
  const banner = document.getElementById('alertBanner');
  const text   = document.getElementById('alertText');
  if (!banner || !text) return;
  text.textContent = message;
  banner.classList.add('visible');
  clearTimeout(alertTimeout);
  alertTimeout = setTimeout(() => banner.classList.remove('visible'), ALERT_DURATION);
}

/* ═══════════════════════ CONNECTION UI ════════════════════ */
function setConnectionState(state) {
  const badge = document.getElementById('systemBadge');
  const text  = document.getElementById('systemStatusText');
  if (!badge || !text) return;
  badge.className = 'system-badge ' + state;
  text.textContent = state === 'connected' ? 'SYSTEM ONLINE' : 'CONNECTING…';
}

/* ═══════════════════════ CLOCK & UPTIME ═══════════════════ */
function updateClock() {
  const now = new Date();
  const time = now.toTimeString().slice(0, 8);
  const el = document.getElementById('clockDisplay');
  if (el) el.textContent = time;

  const elapsed = Math.floor((Date.now() - uptimeStart) / 1000);
  const mm = String(Math.floor(elapsed / 60)).padStart(2, '0');
  const ss = String(elapsed % 60).padStart(2, '0');
  const up = document.getElementById('statUptime');
  if (up) up.textContent = `${mm}:${ss}`;
}
setInterval(updateClock, 1000);
updateClock();

/* ═══════════════════════ MANUAL TEST ══════════════════════ */
function testOverride(direction, zone) {
  fetch('/api/override', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ direction, zone }),
  }).then(r => r.json()).then(d => {
    if (d.ok) log(`Test triggered: ${direction} → ${zone}`, zone === 'safe' ? 'safe' : 'critical');
  }).catch(e => log('Override error: ' + e.message, 'caution'));
}

/* ═══════════════════════ INIT ═════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  initCamRefs();
  log('BlindSpotGuard Dashboard loaded.', 'info');
  log('Attempting WebSocket connection…', 'info');
});
