import {nearestIndex} from './math.mjs';
import {FramePreview, VideoPreview, canPlayOriginal, canPlayPreview, configureLensSelect, lensMedia, secondsInVideo} from './media.mjs';

// 第二路视频跟随主时间轴；画面解码不反过来改变左右手轨迹的时间。
export class CompanionVideo {
  constructor(root) {
    this.root = root; this.video = root.querySelector('video'); this.note = root.querySelector('[data-note]');
    this.frames = new FramePreview(root.querySelector('img'), () => {}, error => { this.note.textContent = error.message; });
    this.preview = new VideoPreview(url => {
      if (canPlayPreview(this.video)) this.setSource(url);
      else if (this.lens === 'rear') this.fallback();
    }, error => {
      if (this.lens === 'rear') this.note.textContent = error.message;
    });
    this.lensSelect = root.querySelector('[data-lens]');
    this.lensSelect?.addEventListener('change', () => this.selectLens(this.lensSelect.value));
    const resumeSync = () => this.applyTarget();
    this.video.addEventListener('loadedmetadata', resumeSync);
    this.video.addEventListener('seeked', resumeSync);
    this.video.addEventListener('error', () => { if (this.recording && this.video.getAttribute('src')) this.fallback(); });
  }
  reset(recording = null) {
    this.recording = recording;
    this.root.hidden = !recording;
    this.time = recording?.times[0] ?? 0; this.playing = false; this.rate = 1;
    if (this.lensSelect) configureLensSelect(this.lensSelect, recording, recording?.arm);
    if (recording) this.root.querySelector('[data-arm]').textContent = recording.arm === 'right' ? '右手' : '左手';
    this.selectLens('front');
  }
  selectLens(lens) {
    const media = lensMedia(this.recording, lens);
    if (lens === 'rear' && !media) return;
    this.media = media; this.lens = lens;
    this.generation = (this.generation || 0) + 1; this.playRequest = null;
    this.preview.reset(); this.frames.reset();
    this.frameMode = false; this.video.hidden = false; this.video.pause();
    this.video.removeAttribute('src'); this.video.poster = media?.poster_url || ''; this.video.load();
    if (!media) return;
    if (lens === 'rear' && !media.video_url) {
      this.note.textContent = '正在准备环境镜头';
      this.preview.reset(media.preview_url);
      return;
    }
    this.note.textContent = media.video_url ? '' : '未保留视频';
    const webm = canPlayPreview(this.video);
    const ready = media.preview_video_url && webm;
    this.setSource(ready ? media.preview_video_url : media.video_url);
    if (!ready && media.preview_url && webm) this.preview.reset(media.preview_url);
    if (!ready && !canPlayOriginal(this.video)) this.fallback();
  }
  setSource(url) {
    if (!this.recording || !url) return;
    this.generation++; this.playRequest = null;
    this.frames.reset(); this.frameMode = false; this.video.hidden = false;
    this.video.src = url; this.video.load(); this.note.textContent = '';
  }
  fallback() {
    if (!this.media?.frame_url || this.frameMode) return;
    this.frameMode = true; this.video.pause(); this.video.removeAttribute('src'); this.video.load(); this.video.hidden = true;
    this.frames.reset(this.media.frame_url); this.note.textContent = '兼容画面 · 跟随任务时间';
    this.synchronize(this.time, this.playing, this.rate);
  }
  synchronize(time, playing = false, rate = 1) {
    this.time = time; this.playing = playing; this.rate = rate;
    this.applyTarget();
  }
  applyTarget() {
    if (!this.recording) return;
    const {time, playing, rate} = this;
    const seconds = secondsInVideo(this.recording, time);
    if (this.frameMode) {
      const index = nearestIndex(this.recording.times, time);
      const frameSeconds = secondsInVideo(this.recording, this.recording.times[index]);
      this.frames.request(index, frameSeconds); return;
    }
    const video = this.video;
    if (!playing && !video.paused) video.pause();
    if (video.readyState < 1 || video.seeking) return;
    video.playbackRate = rate;
    const target = Number.isFinite(video.duration) ? Math.min(seconds, video.duration) : seconds;
    if (Math.abs(video.currentTime - target) > (playing ? .15 : .001)) {
      video.currentTime = target;
    }
    if (playing && video.paused && !this.playRequest) {
      const generation = this.generation;
      this.playRequest = video.play().catch(error => {
        if (generation === this.generation && this.playing && error.name !== 'AbortError') this.fallback();
      }).finally(() => {
        if (generation === this.generation) this.playRequest = null;
      });
    }
  }
}
