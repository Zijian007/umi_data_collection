import {clamp} from './math.mjs';

export class SpeedChart {
  constructor(canvas, onTime) {
    this.canvas = canvas; this.ctx = canvas.getContext('2d'); this.index = 0;
    new ResizeObserver(() => this.draw()).observe(canvas);
    canvas.addEventListener('pointerdown', event => {
      if (!this.times) return;
      const box = canvas.getBoundingClientRect();
      onTime(this.times[0] + clamp((event.clientX - box.left - 45) / (box.width - 64), 0, 1) * (this.times.at(-1) - this.times[0]));
    });
  }
  load(times, values) { this.times = times; this.values = values; this.draw(); }
  draw() {
    const box = this.canvas.getBoundingClientRect(); if (!box.width || !box.height) return;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(box.width * dpr); this.canvas.height = Math.round(box.height * dpr);
    const ctx = this.ctx; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = box.width, h = box.height, left = 45, right = w - 19, top = 17, bottom = h - 25;
    const max = this.values ? this.values.reduce((max, value) => Math.max(max, value), .02) * 1.15 : .5;
    ctx.font = '9px ui-monospace,monospace'; ctx.fillStyle = '#657e78'; ctx.lineWidth = 1;
    for (let n = 0; n <= 2; n++) {
      const y = bottom - (bottom - top) * n / 2;
      ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.strokeStyle = '#2e424030'; ctx.stroke();
      ctx.fillText((max * n / 2).toFixed(2), 10, y + 3);
    }
    if (!this.times) return;
    const duration = this.times.at(-1) - this.times[0], sx = time => left + (time - this.times[0]) / duration * (right - left), sy = value => bottom - value / max * (bottom - top);
    for (let n = 0; n <= 5; n++) ctx.fillText((this.times[0] + duration * n / 5).toFixed(1) + 's', left + (right - left) * n / 5 - 10, h - 8);
    const stride = Math.max(1, Math.ceil(this.times.length / 2000));
    const path = () => { ctx.beginPath(); this.times.forEach((time, i) => { if (i % stride !== 0 && i !== this.times.length - 1) return; if (!i) ctx.moveTo(sx(time), sy(this.values[i])); else ctx.lineTo(sx(time), sy(this.values[i])); }); };
    path(); ctx.lineTo(right, bottom); ctx.lineTo(left, bottom); ctx.closePath();
    const gradient = ctx.createLinearGradient(0, top, 0, bottom); gradient.addColorStop(0, '#8ee3bf25'); gradient.addColorStop(1, '#8ee3bf00'); ctx.fillStyle = gradient; ctx.fill();
    path(); ctx.strokeStyle = '#86c6ac'; ctx.lineWidth = 1.3; ctx.stroke();
    const x = sx(this.times[this.index]), y = sy(this.values[this.index]);
    ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.strokeStyle = '#b7edcf70'; ctx.setLineDash([2, 3]); ctx.stroke(); ctx.setLineDash([]);
    ctx.beginPath(); ctx.arc(x, y, 3.5, 0, Math.PI * 2); ctx.fillStyle = '#d3f5e4'; ctx.fill();
  }
}
