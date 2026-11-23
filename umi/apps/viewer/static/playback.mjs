import {nearestIndex, clamp} from './math.mjs';
import {CompanionVideo} from './paired.mjs';
import {
  FramePreview, VideoPreview, canPlayOriginal, canPlayPreview, configureLensSelect, lensMedia, secondsInVideo, videoTimeOffset,
} from './media.mjs';

// Own the task clock and both video streams. The page renders the selected
// trajectory sample when a compatible image is ready or native playback advances.
export function createPlayback($, {renderFrame, message}) {
  const video = $('video');
  const companionVideo = new CompanionVideo($('companion-video'));
  const state = {
    recording: null, index: 0, playTime: 0, lastTick: 0, playing: false, loop: false,
    sourceGeneration: 0, videoReady: false, frameMode: false, videoSource: null, manualFrameMode: false,
  };
  const videoSeconds = time => secondsInVideo(state.recording, time);
  const playbackRate = () => Number($('rate').value);
  const framePreview = new FramePreview($('video-frame'), index => {
    $('video-empty').hidden = true;
    displayFrame(index, true);
  }, error => {
    pause(); $('video-empty').hidden = false;
    $('video-empty').querySelector('p').textContent = `兼容播放失败：${error.message}`;
    message(`兼容播放失败：${error.message}`, true);
  });
  const videoPreview = new VideoPreview(url => {
    if (state.activeLens === 'rear') {
      $('compatibility-mode').disabled = false;
      if (!canPlayPreview(video)) { enableFrameMode(); return; }
    }
    if (!state.manualFrameMode && url !== state.videoSource) showVideo(url, true);
  }, error => {
    if (state.activeLens === 'rear') {
      $('video-empty').querySelector('p').textContent = error.message;
      $('video-empty').hidden = false;
    } else {
      $('video-note').textContent = state.frameMode ? '兼容播放 · 流畅预览暂不可用' : '原视频播放 · 流畅预览暂不可用';
    }
  });

  function load(recording) {
    state.recording = recording;
    state.index = 0;
    state.playTime = recording.times[0];
    companionVideo.reset(recording.companion);
    configureLensSelect($('primary-lens'), recording, recording.primary_arm);
    selectLens('front');
  }

  function displayFrame(index, imageReady = false) {
    const recording = state.recording;
    if (!recording) return;
    index = clamp(Math.round(index), 0, recording.times.length - 1);
    if (state.frameMode && !imageReady && framePreview.displayed !== index) {
      framePreview.request(index, videoSeconds(recording.times[index]));
      return;
    }
    state.index = index;
    renderFrame(index);
    companionVideo.synchronize(recording.times[index], state.playing, playbackRate());
  }

  function toggle() {
    return state.playing ? pause() : play();
  }

  function step(delta) {
    seek(state.index + delta);
  }

  function selectLens(lens) {
    const recording = state.recording;
    const media = lensMedia(recording, lens);
    if (!media) return;
    state.sourceGeneration++;
    state.activeLens = lens; state.media = media; state.manualFrameMode = false;
    videoPreview.reset(); framePreview.reset();
    state.frameMode = false; state.videoReady = false; state.videoSource = null;
    video.pause(); video.removeAttribute('src'); video.hidden = false;
    video.poster = media.poster_url || ''; video.load();
    $('compatibility-mode').textContent = '兼容播放';
    $('compatibility-mode').disabled = !media.frame_url || !media.video_url;
    $('video-empty').hidden = Boolean(media.poster_url);
    $('video-note').textContent = '';
    if (lens === 'rear' && !media.video_url) {
      $('video-empty').querySelector('p').textContent = '正在准备环境镜头…';
      videoPreview.reset(media.preview_url);
      return;
    }
    $('video-empty').querySelector('p').textContent = media.video_url ? '正在加载画面…' : media.video_note;
    const webm = canPlayPreview(video);
    const ready = media.preview_video_url && webm;
    if (media.video_url) showVideo(ready ? media.preview_video_url : media.video_url, Boolean(ready));
    if (!ready && media.preview_url && webm) videoPreview.reset(media.preview_url);
    if (!ready && media.video_url && !canPlayOriginal(video)) enableFrameMode();
  }

  function showVideo(url, preview = false) {
    state.sourceGeneration++;
    framePreview.reset(); state.frameMode = false; state.videoReady = false; state.videoSource = url;
    video.pause(); video.hidden = false; video.preload = preview ? 'auto' : 'metadata';
    video.src = url; video.load();
    $('video-empty').hidden = Boolean(state.media.poster_url);
    $('compatibility-mode').textContent = '兼容播放'; $('compatibility-mode').disabled = !state.media.frame_url;
    $('video-note').textContent = preview ? '流畅预览' : '';
  }

  function seek(index) {
    if (!state.recording) return;
    index = clamp(Math.round(index), 0, state.recording.times.length - 1);
    state.playTime = state.recording.times[index];
    pause();
    displayFrame(index);
    if (state.videoReady) video.currentTime = videoSeconds(state.playTime);
  }
  function pause() {
    state.playing = false; video.pause();
    companionVideo.synchronize(state.playTime, false, playbackRate());
    $('play').textContent = '▶'; $('play').setAttribute('aria-label', '播放');
  }
  async function play() {
    if (!state.recording) return;
    if (state.index >= state.recording.times.length - 1) seek(0);
    if (!state.frameMode) state.playTime = state.recording.times[state.index];
    state.lastTick = performance.now();
    state.playing = true; $('play').textContent = 'Ⅱ'; $('play').setAttribute('aria-label', '暂停');
    companionVideo.synchronize(state.playTime, true, playbackRate());
    if (state.videoReady) {
      video.currentTime = videoSeconds(state.playTime);
      video.playbackRate = playbackRate();
      await resumeVideo();
    }
  }
  async function resumeVideo() {
    const generation = state.sourceGeneration;
    try { await video.play(); } catch (error) {
      if (generation === state.sourceGeneration && state.playing && error.name !== 'AbortError') {
        if (!enableFrameMode()) pause();
      }
    }
  }
  function tick(now) {
    if (state.playing && state.recording) {
      const times = state.recording.times;
      if (state.videoReady) state.playTime = videoTimeOffset(state.recording) + video.currentTime;
      else state.playTime += Math.min((now - state.lastTick) / 1000, .2) * playbackRate();
      if (state.playTime >= times.at(-1)) {
        if (state.loop) {
          state.playTime = times[0];
          if (state.videoReady) {
            video.currentTime = videoSeconds(times[0]);
            resumeVideo();
          }
        }
        else { displayFrame(times.length - 1); pause(); }
      }
      const index = nearestIndex(times, state.playTime);
      if (index !== state.index) displayFrame(index);
    }
    state.lastTick = now; requestAnimationFrame(tick);
  }
  video.addEventListener('loadedmetadata', () => {
    if (state.frameMode || !state.videoSource || !video.currentSrc.endsWith(state.videoSource)) return;
    state.videoReady = true; $('video-empty').hidden = true;
    video.playbackRate = playbackRate();
    video.currentTime = videoSeconds(state.playTime);
    displayFrame(nearestIndex(state.recording.times, state.playTime));
    if (state.playing) resumeVideo();
  });
  function enableFrameMode() {
    if (state.frameMode) return true;
    if (!state.media?.frame_url) {
      $('video-empty').hidden = false;
      $('video-empty').querySelector('p').textContent = '当前服务未提供兼容播放，请重启 umi-view 并刷新页面。';
      return false;
    }
    state.sourceGeneration++;
    state.frameMode = true; state.videoReady = false; state.videoSource = null;
    video.pause(); video.removeAttribute('src'); video.load(); video.hidden = true;
    framePreview.reset(state.media.frame_url);
    $('compatibility-mode').textContent = '兼容模式'; $('compatibility-mode').disabled = true;
    $('video-empty').hidden = false;
    $('video-empty').querySelector('p').textContent = '正在加载兼容画面…';
    $('video-note').textContent = '兼容播放 · 视频画面与轨迹同步';
    message('已启用视频兼容播放；画面按服务器可达帧率显示，轨迹与画面保持对应。');
    displayFrame(nearestIndex(state.recording.times, state.playTime));
    return true;
  }
  video.addEventListener('error', () => {
    if (!state.videoSource || !video.getAttribute('src') || state.frameMode) return;
    enableFrameMode();
  });
  $('primary-lens').addEventListener('change', event => selectLens(event.target.value));
  $('compatibility-mode').addEventListener('click', () => { state.manualFrameMode = true; enableFrameMode(); });
  video.addEventListener('ended', () => { if (!state.recording) return; if (state.loop) { seek(0); play(); } else { seek(state.recording.times.length - 1); } });
  $('play').addEventListener('click', () => toggle());
  $('timeline').addEventListener('input', event => seek(Number(event.target.value)));
  $('previous-frame').addEventListener('click', () => step(-1));
  $('next-frame').addEventListener('click', () => step(1));
  $('rate').addEventListener('change', () => {
    const rate = playbackRate();
    video.playbackRate = rate;
    companionVideo.synchronize(state.playTime, state.playing, rate);
  });
  $('loop').addEventListener('click', () => { state.loop = !state.loop; $('loop').setAttribute('aria-pressed', String(state.loop)); });
  $('video-fullscreen').addEventListener('click', async () => {
    if (!state.recording) return;
    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
      } else if (state.recording.companion) {
        await document.documentElement.requestFullscreen();
      } else if (state.videoReady || state.frameMode) {
        await (state.frameMode ? video.parentElement : video).requestFullscreen();
      }
    } catch (error) {
      message(`无法进入全屏：${error.message}`, true);
    }
  });
  document.addEventListener('fullscreenchange', () => {
    const paired = Boolean(state.recording?.companion);
    const dualFullscreen = paired && document.fullscreenElement === document.documentElement;
    document.body.classList.toggle('dual-fullscreen', dualFullscreen);
    $('video-fullscreen').textContent = dualFullscreen ? '退出全屏' : paired ? '双路全屏' : '全屏';
  });

  requestAnimationFrame(tick);
  return {load, pause, seek, step, toggle, redraw() { displayFrame(state.index); }};
}
