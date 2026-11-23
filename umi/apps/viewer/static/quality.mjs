import {clamp} from './math.mjs';

export const qualityLabels = {
  visual: '视觉定位', map: '地图定位', natural: '自然特征', visual_and_poster: '视觉＋海报',
  poster: '海报', rot: '仅朝向', imu: '仅 IMU', unknown: '未知',
};

export function frameQuality(frame, summary = {}) {
  const visual = summary.reconstruction_backend === 'poster_vio' ? 'natural'
    : summary.mode === 'scene' ? 'map' : 'visual';
  const source = frame?.position_source;
  if (source === 'visual') return visual;
  if (source === 'poster' || source === 'visual_and_poster') return source;
  if (frame?.constraint === 'rot') return 'rot';
  if (source === 'imu' || frame?.constraint === 'imu') return 'imu';
  return frame?.constraint === 'full' ? visual : 'unknown';
}

export function qualitySegments(times, frames, summary, start, end) {
  const segments = [];
  if (end <= start) return segments;
  for (const [i, time] of times.entries()) {
    const from = Math.max(start, i ? (times[i - 1] + time) / 2 : times[0]);
    const to = Math.min(end, i + 1 < times.length ? (time + times[i + 1]) / 2 : times.at(-1));
    if (to <= from) continue;
    const kind = frameQuality(frames[i], summary);
    const previous = segments.at(-1);
    if (previous?.kind === kind) previous.end = to;
    else segments.push({start: from, end: to, kind});
  }
  return segments;
}

export class QualityTimeline {
  constructor(root, onSeek) {
    this.root = root;
    this.onSeek = onSeek;
    this.legend = root.querySelector('[data-quality-legend]');
    this.readout = root.querySelector('[data-quality-readout]');
    this.rows = [...root.querySelectorAll('[data-quality-arm]')];
  }

  load(recording) {
    this.root.hidden = !recording.companion;
    if (this.root.hidden) return;
    this.start = recording.times[0];
    this.duration = recording.times.at(-1) - this.start;
    this.readout.textContent = '';
    const hands = {[recording.primary_arm]: recording, [recording.companion.arm]: recording.companion};
    const kinds = new Set();
    for (const row of this.rows) {
      const arm = row.dataset.qualityArm;
      const hand = hands[arm];
      if (!hand) continue;
      const label = arm === 'left' ? '左手' : '右手';
      const track = row.querySelector('button');
      const segments = qualitySegments(hand.times, hand.frames || [], hand.summary, this.start, this.start + this.duration);
      track.replaceChildren();
      for (const segment of segments) {
        const span = document.createElement('span');
        span.dataset.kind = segment.kind;
        span.style.width = `${(segment.end - segment.start) / this.duration * 100}%`;
        span.title = this.describe(label, segment);
        track.append(span);
        kinds.add(segment.kind);
      }
      let active = null;
      track.onpointermove = event => {
        const time = this.timeAt(track, event.clientX);
        const segment = segments.find(({end}) => time < end) || segments.at(-1);
        if (segment) this.readout.textContent = this.describe(label, segment);
      };
      track.onpointerleave = () => { this.readout.textContent = ''; };
      track.onblur = () => { active = null; this.readout.textContent = ''; };
      track.onclick = event => {
        if (event.detail > 0) {
          this.onSeek(this.timeAt(track, event.clientX));
        } else if (segments.length) {
          const current = segments.find(({end}) => this.time < end) || segments.at(-1);
          const segment = active == null ? current : segments[active];
          this.onSeek((segment.start + segment.end) / 2);
        }
      };
      track.onkeydown = event => {
        if (!['ArrowLeft', 'ArrowRight'].includes(event.key) || !segments.length) return;
        event.preventDefault();
        event.stopPropagation();
        const current = segments.findIndex(({end}) => this.time < end);
        active = clamp((active ?? Math.max(0, current)) + (event.key === 'ArrowLeft' ? -1 : 1), 0, segments.length - 1);
        this.readout.textContent = this.describe(label, segments[active]);
      };
    }
    this.legend.replaceChildren();
    for (const [kind, label] of Object.entries(qualityLabels)) {
      if (!kinds.has(kind)) continue;
      const item = document.createElement('span');
      const swatch = document.createElement('i');
      swatch.dataset.kind = kind;
      item.append(swatch, label);
      this.legend.append(item);
    }
    this.select(this.start);
  }

  timeAt(track, x) {
    const {left, width} = track.getBoundingClientRect();
    return this.start + this.duration * clamp((x - left) / width, 0, 1);
  }

  describe(label, {start, end, kind}) {
    return `${label} · ${qualityLabels[kind]} · ${(start - this.start).toFixed(2)}–${(end - this.start).toFixed(2)}s · ${(end - start).toFixed(2)}s`;
  }

  select(time) {
    this.time = time;
    if (!this.root.hidden) {
      this.root.style.setProperty('--quality-cursor', `${clamp((time - this.start) / this.duration, 0, 1) * 100}%`);
    }
  }
}
