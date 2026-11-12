import {nearestIndex} from './math.mjs';
import {FramePreview, VideoPreview} from './media.mjs';

// 第二路视频跟随主时间轴；画面解码不反过来改变左右手轨迹的时间。
export class CompanionVideo {
  constructor(root) {
    this.root = root; this.video = root.querySelector('video'); this.note = root.querySelector('[data-note]');
    this.frames = new FramePreview(root.querySelector('img'), () => {}, error => { this.note.textContent = error.message; });
    this.preview = new VideoPreview(url => this.setSource(url), () => {});
    this.video.addEventListener('loadedmetadata', () => this.synchronize(this.time, this.playing, this.rate));
    this.video.addEventListener('error', () => { if (this.recording && this.video.getAttribute('src')) this.fallback(); });
  }
  reset(recording = null) {
    this.preview.reset(); this.frames.reset(); this.video.pause(); this.video.removeAttribute('src'); this.video.load();
    this.recording = recording; this.frameMode = false; this.root.hidden = !recording;
    this.time = recording?.times[0] ?? 0; this.playing = false; this.rate = 1;
    if (!recording) return;
    this.root.querySelector('[data-arm]').textContent = recording.arm === 'right' ? '右手工作镜头' : '左手工作镜头';
    this.video.poster = recording.poster_url || '';
    const ready = recording.preview_video_url && this.video.canPlayType('video/webm; codecs="vp8"');
    this.setSource(ready ? recording.preview_video_url : recording.video_url);
    if (!ready && recording.preview_url && this.video.canPlayType('video/webm; codecs="vp8"')) this.preview.reset(recording.preview_url);
    if (!ready && !this.video.canPlayType('video/mp4; codecs="avc1.640020"')) this.fallback();
  }
  setSource(url) {
    if (!this.recording || !url) return;
    this.frames.reset(); this.frameMode = false; this.video.hidden = false;
    this.video.src = url; this.video.load(); this.note.textContent = '与双手任务时间同步';
  }
  fallback() {
    if (!this.recording?.frame_url || this.frameMode) return;
    this.frameMode = true; this.video.pause(); this.video.removeAttribute('src'); this.video.load(); this.video.hidden = true;
    this.frames.reset(this.recording.frame_url); this.note.textContent = '兼容画面 · 跟随任务时间';
    this.synchronize(this.time, this.playing, this.rate);
  }
  synchronize(time, playing = false, rate = 1) {
    this.time = time; this.playing = playing; this.rate = rate;
    if (!this.recording) return;
    const seconds = Math.max(0, time - (this.recording.video_time_offset_s ?? this.recording.times[0]));
    if (this.frameMode) {
      const index = nearestIndex(this.recording.times, time);
      const frameSeconds = Math.max(0, this.recording.times[index] - (this.recording.video_time_offset_s ?? this.recording.times[0]));
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
