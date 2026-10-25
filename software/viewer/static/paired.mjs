import {nearestIndex} from './math.mjs';
import {FramePreview, VideoPreview} from './media.mjs';

// 第二路视频跟随主时间轴；画面解码不反过来改变左右手轨迹的时间。
export class CompanionVideo {
  constructor(root) {
    this.root = root; this.video = root.querySelector('video'); this.note = root.querySelector('[data-note]');
    this.frames = new FramePreview(root.querySelector('img'), () => {}, error => { this.note.textContent = error.message; });
    this.preview = new VideoPreview(url => this.setSource(url), () => {});
    this.video.addEventListener('loadedmetadata', () => this.synchronize(this.time, this.playing, this.rate));
    this.video.addEventListener('error', () => { if (this.data && this.video.getAttribute('src')) this.fallback(); });
  }
  reset(data = null) {
    this.preview.reset(); this.frames.reset(); this.video.pause(); this.video.removeAttribute('src'); this.video.load();
    this.data = data; this.frameMode = false; this.root.hidden = !data;
    this.time = data?.times[0] ?? 0; this.playing = false; this.rate = 1;
    if (!data) return;
    this.root.querySelector('[data-arm]').textContent = data.arm === 'right' ? '右手前镜头' : '左手前镜头';
    this.video.poster = data.poster_url || '';
    const ready = data.preview_video_url && this.video.canPlayType('video/webm; codecs="vp8"');
    this.setSource(ready ? data.preview_video_url : data.video_url);
    if (!ready && data.preview_url && this.video.canPlayType('video/webm; codecs="vp8"')) this.preview.reset(data.preview_url);
    if (!ready && !this.video.canPlayType('video/mp4; codecs="avc1.640020"')) this.fallback();
  }
  setSource(url) {
    if (!this.data || !url) return;
    this.frames.reset(); this.frameMode = false; this.video.hidden = false;
    this.video.src = url; this.video.load(); this.note.textContent = '与双手任务时间同步';
  }
  fallback() {
    if (!this.data?.frame_url || this.frameMode) return;
    this.frameMode = true; this.video.pause(); this.video.removeAttribute('src'); this.video.load(); this.video.hidden = true;
    this.frames.reset(this.data.frame_url); this.note.textContent = '兼容画面 · 跟随任务时间';
    this.synchronize(this.time, this.playing, this.rate);
  }
  synchronize(time, playing = false, rate = 1) {
    this.time = time; this.playing = playing; this.rate = rate;
    if (!this.data) return;
    const seconds = Math.max(0, time - (this.data.video_time_offset_s ?? this.data.times[0]));
    if (this.frameMode) {
      const index = nearestIndex(this.data.times, time);
      const frameSeconds = Math.max(0, this.data.times[index] - (this.data.video_time_offset_s ?? this.data.times[0]));
      this.frames.request(index, frameSeconds); return;
    }
    const video = this.video;
    if (video.readyState < 1) return;
    video.playbackRate = rate;
    if (!video.seeking && Math.abs(video.currentTime - seconds) > (playing ? .15 : .001)) video.currentTime = seconds;
    if (playing && video.paused) video.play().catch(() => this.fallback());
    else if (!playing && !video.paused) video.pause();
  }
}
