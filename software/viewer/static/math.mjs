export function nearestIndex(times, time) {
  if (!times.length) return 0;
  let lo = 0, hi = times.length - 1;
  while (lo < hi) { const mid = (lo + hi) >>> 1; if (times[mid] < time) lo = mid + 1; else hi = mid; }
  return lo > 0 && time - times[lo - 1] <= times[lo] - time ? lo - 1 : lo;
}

export function rotateQuaternion(vector, q) {
  const [x, y, z, w] = q, [vx, vy, vz] = vector;
  const tx = 2 * (y * vz - z * vy), ty = 2 * (z * vx - x * vz), tz = 2 * (x * vy - y * vx);
  return [vx + w * tx + y * tz - z * ty, vy + w * ty + z * tx - x * tz, vz + w * tz + x * ty - y * tx];
}

export function formatTime(value) {
  const t = Math.max(0, Number(value) || 0);
  return `${String(Math.floor(t / 60)).padStart(2, '0')}:${(t % 60).toFixed(2).padStart(5, '0')}`;
}

export function niceStep(value) {
  const scale = 10 ** Math.floor(Math.log10(Math.max(value, 1e-8)));
  const n = value / scale;
  return (n < 1.5 ? 1 : n < 3.5 ? 2 : n < 7.5 ? 5 : 10) * scale;
}

export const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
