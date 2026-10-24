// Single-flight JPEG preview for browsers without native video codec support.
// While decoding, coalesce requests to the latest desired time rather than queue
// every trajectory frame. Present each image together with its matching pose.
export class FramePreview {
  constructor(image, onFrame, onError, {fetcher = (...args) => fetch(...args), decode = decodeImage} = {}) {
    this.image = image; this.onFrame = onFrame; this.onError = onError;
    this.fetcher = fetcher; this.decode = decode; this.generation = 0;
    this.lastStart = -Infinity; this.busy = false; this.pending = null; this.objectURL = null;
  }
  reset(url = null) {
    this.generation++; this.controller?.abort(); clearTimeout(this.timer);
    if (this.objectURL) URL.revokeObjectURL(this.objectURL);
    this.url = url; this.objectURL = null; this.pending = null; this.busy = false;
    this.displayed = null; this.lastStart = -Infinity;
    this.image.removeAttribute('src'); this.image.hidden = true;
  }
  request(index, seconds) {
    if (!this.url) return;
    if (this.displayed === index && !this.busy) { this.pending = null; clearTimeout(this.timer); return; }
    this.pending = {index, seconds}; this.schedule();
  }
  schedule() {
    if (this.busy || !this.pending || !this.url) return;
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this.pump(), Math.max(0, 125 - (performance.now() - this.lastStart)));
  }
  async pump() {
    if (this.busy || !this.pending || !this.url) return;
    const frame = this.pending, generation = this.generation;
    this.pending = null; this.busy = true; this.lastStart = performance.now();
    this.controller = new AbortController(); let objectURL = null;
    try {
      const url = `${this.url}?t=${frame.seconds.toFixed(6)}`;
      const response = await this.fetcher(url, {signal: this.controller.signal, cache: 'no-store'});
      if (!response.ok) { const body = await response.json(); throw new Error(body.error || `画面读取失败（${response.status}）`); }
      objectURL = URL.createObjectURL(await response.blob());
      await this.decode(objectURL);
      if (generation !== this.generation) return;
      const previous = this.objectURL;
      this.image.src = objectURL; this.image.hidden = false;
      this.objectURL = objectURL; objectURL = null; this.displayed = frame.index;
      if (previous) URL.revokeObjectURL(previous);
      this.onFrame(frame.index);
    } catch (error) {
      if (generation === this.generation && error.name !== 'AbortError') {
        this.pending = null; this.url = null; this.onError(error);
      }
    } finally {
      if (objectURL) URL.revokeObjectURL(objectURL);
      if (generation === this.generation) {
        this.busy = false;
        if (this.pending?.index === this.displayed) this.pending = null;
        this.schedule();
      }
    }
  }
}

function decodeImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve();
    image.onerror = () => reject(new Error('浏览器无法显示兼容模式的画面。'));
    image.src = url;
  });
}
