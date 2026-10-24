import {clamp, niceStep, rotateQuaternion} from './math.mjs';

const COLORS = {tcp: '#8ee3bf', camera: '#91b7f4', imu: '#d9b67b'};
const add = (a, b) => a.map((v, i) => v + b[i]);

// A small Z-up perspective renderer: no remote assets, WebGL or build step required.
export class TrajectoryScene {
  constructor(canvas, onPick) {
    this.canvas = canvas; this.ctx = canvas.getContext('2d'); this.onPick = onPick;
    this.data = null; this.track = 'tcp'; this.index = 0; this.pose = true; this.overlay = false;
    this.azimuth = -1.0; this.elevation = .55; this.zoom = 1; this.pan = [0, 0]; this.center = [0, 0, 0]; this.span = 1; this.floor = -.15;
    new ResizeObserver(() => this.draw()).observe(canvas);
    canvas.addEventListener('contextmenu', e => e.preventDefault());
    canvas.addEventListener('wheel', e => { e.preventDefault(); this.zoom = clamp(this.zoom * Math.exp(-e.deltaY * .001), .15, 12); this.draw(); }, {passive: false});
    canvas.addEventListener('pointerdown', e => {
      canvas.setPointerCapture(e.pointerId);
      this.drag = {x: e.clientX, y: e.clientY, startX: e.clientX, startY: e.clientY, pan: e.button === 2 || e.shiftKey, moved: false};
    });
    canvas.addEventListener('pointermove', e => {
      if (!this.drag) return;
      const dx = e.clientX - this.drag.x, dy = e.clientY - this.drag.y;
      this.drag.moved ||= Math.hypot(e.clientX - this.drag.startX, e.clientY - this.drag.startY) > 4;
      if (this.drag.pan) { this.pan[0] += dx; this.pan[1] += dy; }
      else { this.azimuth -= dx * .008; this.elevation = clamp(this.elevation + dy * .006, -1.5, 1.55); }
      this.drag.x = e.clientX; this.drag.y = e.clientY; this.draw();
    });
    canvas.addEventListener('pointerup', e => {
      if (this.drag && !this.drag.moved && e.button === 0) this.pick(e);
      this.drag = null;
    });
    canvas.addEventListener('pointercancel', () => { this.drag = null; });
    canvas.addEventListener('dblclick', () => this.view('home'));
  }
  load(data, track = 'tcp') {
    this.data = data; this.track = track; this.index = 0;
    const p = data.tracks[track].position;
    const min = [0, 1, 2].map(k => p.reduce((min, v) => Math.min(min, v[k]), Infinity));
    const max = [0, 1, 2].map(k => p.reduce((max, v) => Math.max(max, v[k]), -Infinity));
    this.center = min.map((v, k) => (v + max[k]) / 2);
    this.span = Math.max(...min.map((v, k) => max[k] - v), .08); this.floor = min[2] - this.span * .12;
    this.view('home');
  }
  view(name) {
    const views = {home: [-1.0, .55], top: [-Math.PI / 2, Math.PI / 2 - .001], front: [-Math.PI / 2, 0], side: [0, 0]};
    [this.azimuth, this.elevation] = views[name] || views.home;
    this.pan = [0, 0]; this.zoom = 1; this.draw();
  }
  project(point) {
    const p = point.map((v, k) => v - this.center[k]);
    const ca = Math.cos(this.azimuth), sa = Math.sin(this.azimuth), ce = Math.cos(this.elevation), se = Math.sin(this.elevation);
    const x = -sa * p[0] + ca * p[1];
    const y = -se * ca * p[0] - se * sa * p[1] + ce * p[2];
    const depth = ce * ca * p[0] + ce * sa * p[1] + se * p[2];
    const distance = this.span * 3.5;
    const scale = Math.min(this.width, this.height) * .72 / this.span * this.zoom * distance / Math.max(distance - depth, this.span * .1);
    return [this.width / 2 + x * scale + this.pan[0], this.height / 2 - y * scale + this.pan[1], depth];
  }
  line(a, b, color, width = 1) {
    const ctx = this.ctx, x = this.project(a), y = this.project(b);
    ctx.beginPath(); ctx.moveTo(x[0], x[1]); ctx.lineTo(y[0], y[1]); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke();
  }
  path(points, end, color, width, dashed = false) {
    const ctx = this.ctx; ctx.beginPath();
    const stride = Math.max(1, Math.ceil(points.length / 6000));
    for (let i = 0; i <= end; i += stride) { const p = this.project(points[i]); if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]); }
    const last = this.project(points[end]); ctx.lineTo(last[0], last[1]);
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    ctx.setLineDash(dashed ? [3, 5] : []); ctx.stroke(); ctx.setLineDash([]);
  }
  grid() {
    const step = niceStep(this.span / 5), half = step * 5;
    const cx = Math.round(this.center[0] / step) * step, cy = Math.round(this.center[1] / step) * step;
    for (let i = -5; i <= 5; i++) {
      const color = i === 0 ? '#3b56544d' : '#36504c35';
      this.line([cx + i * step, cy - half, this.floor], [cx + i * step, cy + half, this.floor], color);
      this.line([cx - half, cy + i * step, this.floor], [cx + half, cy + i * step, this.floor], color);
    }
    this.ctx.font = '9px ui-monospace,monospace'; this.ctx.fillStyle = '#5f8175';
    for (let i = -2; i <= 2; i++) {
      const point = this.project([cx + i * step, cy - half * .6, this.floor]);
      this.ctx.fillText((cx + i * step).toFixed(step < .1 ? 2 : 1), point[0], point[1] + 10);
    }
  }
  triad() {
    const ctx = this.ctx, origin = [this.width - 38, this.height - 40];
    const center = this.project(this.center);
    for (const [axis, color, offset] of [['X', '#d38c80', [1, 0, 0]], ['Y', '#8bd4a1', [0, 1, 0]], ['Z', '#8ab4e0', [0, 0, 1]]]) {
      const p = this.project(add(this.center, offset.map(v => v * this.span * .12)));
      const dx = p[0] - center[0], dy = p[1] - center[1];
      const length = Math.hypot(dx, dy) || 1;
      const x = origin[0] + dx / length * 22, y = origin[1] + dy / length * 22;
      ctx.beginPath(); ctx.moveTo(...origin); ctx.lineTo(x, y); ctx.strokeStyle = color; ctx.lineWidth = 1.2; ctx.stroke();
      ctx.fillStyle = color; ctx.font = '9px monospace'; ctx.fillText(axis, x + 3, y + 3);
    }
  }
  draw() {
    const box = this.canvas.getBoundingClientRect(); if (!box.width || !box.height) return;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    if (this.canvas.width !== Math.round(box.width * dpr) || this.canvas.height !== Math.round(box.height * dpr)) {
      this.canvas.width = Math.round(box.width * dpr); this.canvas.height = Math.round(box.height * dpr);
    }
    this.width = box.width; this.height = box.height;
    const ctx = this.ctx; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, this.width, this.height);
    this.grid();
    if (this.data) {
      const track = this.data.tracks[this.track], position = track.position, index = clamp(this.index, 0, position.length - 1);
      if (this.overlay) for (const [name, other] of Object.entries(this.data.tracks)) if (name !== this.track) this.path(other.position, other.position.length - 1, COLORS[name] + '70', 1, true);
      this.path(position, position.length - 1, COLORS[this.track] + '70', 1.5);
      this.path(position, index, COLORS[this.track], 2.2);
      const first = this.project(position[0]); ctx.beginPath(); ctx.arc(first[0], first[1], 3, 0, Math.PI * 2); ctx.fillStyle = '#658c7b'; ctx.fill();
      if (this.pose) {
        const axis = this.span * .12, q = track.quaternion[index];
        for (const [vector, color] of [[[axis, 0, 0], '#e29a87'], [[0, axis, 0], '#97dba7'], [[0, 0, axis], '#96bff0']]) this.line(position[index], add(position[index], rotateQuaternion(vector, q)), color, 2);
      }
      const current = this.project(position[index]);
      ctx.beginPath(); ctx.arc(current[0], current[1], 9, 0, Math.PI * 2); ctx.fillStyle = '#e4fff018'; ctx.fill();
      ctx.beginPath(); ctx.arc(current[0], current[1], 4, 0, Math.PI * 2); ctx.fillStyle = '#f0fff6'; ctx.shadowColor = '#8ee3bf'; ctx.shadowBlur = 13; ctx.fill(); ctx.shadowBlur = 0;
    }
    this.triad();
  }
  pick(event) {
    if (!this.data) return;
    const box = this.canvas.getBoundingClientRect(), x = event.clientX - box.left, y = event.clientY - box.top;
    let best = Infinity, index = 0;
    this.data.tracks[this.track].position.forEach((point, i) => { const p = this.project(point); const distance = Math.hypot(p[0] - x, p[1] - y); if (distance < best) { best = distance; index = i; } });
    if (best < 24) this.onPick(index);
  }
}
