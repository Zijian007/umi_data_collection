import {nearestIndex, formatTime, clamp} from './math.mjs';
import {TrajectoryScene} from './scene.mjs';
import {SpeedChart} from './chart.mjs';
import {FramePreview, VideoPreview} from './media.mjs';
import {CompanionVideo} from './paired.mjs';
import {tagDistanceReadout} from './measurements.mjs';

const $ = id => document.getElementById(id);
const state = {recording: null, maps: [], catalog: [], root: '', track: 'tcp', index: 0, playing: false, loop: false, load: 0, lastTick: 0, playTime: 0, videoReady: false, frameMode: false, videoSource: null, manualFrameMode: false};
const video = $('video');
const companionVideo = new CompanionVideo($('companion-video'));
const videoSeconds = time => Math.max(0, time - (state.recording.video_time_offset_s ?? state.recording.times[0]));
const scene = new TrajectoryScene($('trajectory-canvas'), index => seek(index));
const chart = new SpeedChart($('speed-canvas'), time => { if (state.recording) seek(nearestIndex(state.recording.times, time)); });
const framePreview = new FramePreview($('video-frame'), index => {
  $('video-empty').hidden = true;
  renderFrame(index, true);
}, error => {
  pause(); $('video-empty').hidden = false;
  $('video-empty').querySelector('p').textContent = `兼容播放失败：${error.message}`;
  message(`兼容播放失败：${error.message}`, true);
});
const videoPreview = new VideoPreview(url => {
  if (!state.manualFrameMode && url !== state.videoSource) showVideo(url, true);
}, () => {
  $('video-note').textContent = state.frameMode ? '兼容播放 · 流畅预览暂不可用' : '原视频播放 · 流畅预览暂不可用';
});
const labels = {full: '地图位置', rot: '仅朝向', imu: 'IMU 衔接', unknown: '未知'};
const colors = {full: '#8ee3bf', rot: '#d9b67b', imu: '#e29a87', unknown: '#849597'};

function message(text = '', error = false) {
  $('message').textContent = text; $('message').className = `message${error ? ' error' : ''}`; $('message').hidden = !text;
}
async function api(url) {
  const response = await fetch(url, {cache: 'no-store'});
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `读取失败（${response.status}）`);
  return body;
}
function recordingLabel(name) {
  const match = name.match(/VID_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/);
  return match ? `${match[1]}.${match[2]}.${match[3]} · ${match[4]}:${match[5]}:${match[6]}` : name;
}
function renderLibrary() {
  const filter = $('search').value.trim().toLowerCase();
  const entries = state.catalog.filter(entry => `${entry.name} ${entry.serial || ''} ${entry.path}`.toLowerCase().includes(filter));
  $('record-count').textContent = state.catalog.length;
  const list = $('recording-list'); list.replaceChildren();
  if (!entries.length) {
    const empty = document.createElement('p'); empty.className = 'empty-list';
    empty.textContent = filter ? '没有匹配的录像。' : '还没有可查看的轨迹。先运行 umi-scene run 处理任务，再刷新此列表。'; list.append(empty); return;
  }
  for (const [index, entry] of entries.entries()) {
    const button = document.createElement('button'); button.className = `recording-item${state.recording?.id === entry.id ? ' active' : ''}`;
    button.setAttribute('aria-current', state.recording?.id === entry.id ? 'true' : 'false'); button.title = entry.reason || entry.path; button.disabled = entry.available === false;
    const top = document.createElement('span'); top.className = 'record-index';
    const title = document.createElement('span'); title.textContent = `TAKE ${String(index + 1).padStart(2, '0')}`;
    const status = document.createElement('i'); status.className = `item-status${['failed', 'rejected_preflight', 'incomplete'].includes(entry.status) ? ' failed' : ''}`;
    top.append(title, status);
    const name = document.createElement('strong'); name.textContent = recordingLabel(entry.name);
    const labels = {pending:'待处理', incomplete:'待补齐', failed:'处理失败', rejected_preflight:'原片检查未通过', ok:'已完成', skipped:'已完成', local_ok:'独立局部轨迹', local_skipped:'独立局部轨迹（复用）'};
    const sub = document.createElement('small'); sub.textContent = `${entry.serial?.slice(-4) || '设备未知'}  /  ${Number.isFinite(entry.duration) ? entry.duration.toFixed(1) + ' s' : '时长未知'}  /  ${entry.rows ?? '—'} 帧 / ${labels[entry.status] || entry.status}${entry.review === 'accepted' ? ' · 已接收' : entry.review === 'rejected' ? ' · 已拒收' : ''}`;
    button.append(top, name, sub); button.addEventListener('click', () => load(entry.id)); list.append(button);
  }
}
async function refresh(initial = false) {
  $('refresh').disabled = true;
  try {
    const catalog = await api('/api/recordings'); state.catalog = catalog.recordings; state.root = catalog.root;
    $('root-path').textContent = catalog.root; $('root-path').title = catalog.root;
    renderLibrary();
    if (initial && catalog.recordings.length) {
      const requested = new URL(location.href).searchParams.get('recording');
      const available = catalog.recordings.filter(item => item.available !== false);
      const entry = available.find(item => item.id === requested) || available[0];
      if (entry) await load(entry.id);
    }
  } catch (error) { message(error.message, true); }
  finally { $('refresh').disabled = false; }
}
async function load(id) {
  const request = ++state.load;
  videoPreview.reset();
  companionVideo.reset();
  pause(); message('正在加载轨迹和视频…'); $('play').disabled = true;
  try {
    const recording = await api(`/api/recordings/${id}`);
    if (request !== state.load) return;
    state.recording = recording; state.track = 'tcp'; state.index = 0; state.videoReady = false; state.frameMode = false; state.playTime = recording.times[0]; state.manualFrameMode = false;
    companionVideo.reset(recording.companion);
    $('video-title').textContent = recording.primary_arm ? (recording.primary_arm === 'left' ? '左手工作镜头' : '右手工作镜头') : '工作镜头';
    video.closest('.video-panel').classList.toggle('paired', Boolean(recording.companion));
    framePreview.reset(); video.hidden = false;
    $('compatibility-mode').textContent = '兼容播放';
    $('compatibility-mode').disabled = !recording.frame_url;
    video.pause(); video.removeAttribute('src'); state.videoSource = null;
    video.poster = recording.poster_url || '';
    const readyPreview = recording.preview_video_url && video.canPlayType('video/webm; codecs="vp8"');
    if (recording.video_url) showVideo(readyPreview ? recording.preview_video_url : recording.video_url, Boolean(readyPreview));
    else video.load();
    $('video-empty').hidden = Boolean(recording.poster_url);
    $('video-empty').querySelector('p').textContent = recording.video_url ? '正在加载工作镜头画面…' : recording.video_note;
    if (!readyPreview) $('video-note').textContent = recording.video_note;
    const summary = recording.summary, selection = summary.calibration_used || {};
    $('record-title').textContent = recordingLabel(recording.name); $('record-title').title = recording.name;
    $('record-meta').textContent = `${recording.name} · ${summary.scene?.frame_id || summary.mode || 'handheld'}`;
    $('status-badge').textContent = summary.mode === 'local' ? '独立回退 · 局部坐标' : summary.status === 'ok' ? '抽取成功' : summary.status === 'failed' ? '抽取失败' : '已加载轨迹';
    $('status-badge').classList.toggle('failed', summary.status === 'failed');
    $('stat-duration').replaceChildren(document.createTextNode((recording.times.at(-1) - recording.times[0]).toFixed(1)), unit(' s'));
    $('detail-serial').textContent = selection.serial || '未记录'; $('detail-calibration').textContent = selection.id || '未记录';
    if (recording.companion) {
      const other = recording.companion.summary.calibration_used || {};
      $('detail-serial').textContent = `L ${selection.serial} / R ${other.serial}`;
      $('detail-calibration').textContent = `L ${selection.id} / R ${other.id}`;
    }
    $('detail-rows').textContent = recording.times.length.toLocaleString();
    $('timeline').max = recording.times.length - 1; $('timeline').disabled = false; $('play').disabled = false;
    $('total-time').textContent = formatTime(recording.times.at(-1) - recording.times[0]);
    $('scene-empty').hidden = true;
    for (const button of document.querySelectorAll('[data-track]')) { button.disabled = !recording.tracks[button.dataset.track]; button.classList.toggle('active', button.dataset.track === 'tcp'); }
    scene.load(recording); chart.load(recording.times, recording.tracks.tcp.speed); setTrack('tcp'); renderLibrary();
    loadMaps(id, request);
    const url = new URL(location.href); url.searchParams.set('recording', id); history.replaceState(null, '', url);
    const warnings = [...recording.notes];
    if (summary.status === 'failed') warnings.unshift(`此结果被标记为失败：${summary.reason || '请查看本次记录'}。轨迹仅供排查。`);
    message(warnings.join(' '), summary.status === 'failed');
    if (!readyPreview && recording.preview_url && video.canPlayType('video/webm; codecs="vp8"')) videoPreview.reset(recording.preview_url);
    if (!readyPreview && recording.video_url && !video.canPlayType('video/mp4; codecs="avc1.640020"')) enableFrameMode();
    document.body.dataset.loaded = 'true';
  } catch (error) {
    if (request === state.load) { message(`加载失败：${error.message}`, true); $('play').disabled = !state.recording; }
  }
}
function unit(text) { const element = document.createElement('small'); element.textContent = text; return element; }
function showVideo(url, preview = false) {
  framePreview.reset(); state.frameMode = false; state.videoReady = false; state.videoSource = url;
  video.pause(); video.hidden = false; video.preload = preview ? 'auto' : 'metadata';
  video.src = url; video.load();
  $('video-empty').hidden = Boolean(state.recording.poster_url);
  $('compatibility-mode').textContent = '兼容播放'; $('compatibility-mode').disabled = !state.recording.frame_url;
  $('video-note').textContent = preview ? '流畅预览 · 与轨迹时间同步' : state.recording.video_note;
}
async function loadMaps(id, request) {
  state.maps = [];
  $('map-select').replaceChildren();
  for (const name of ['map-select', 'show-map', 'show-keyframes', 'show-map-axes']) $(name).disabled = true;
  $('map-status').textContent = '正在读取环境地图…';
  try {
    const mapResponse = await api(`/api/recordings/${id}/map`);
    if (request !== state.load) return;
    state.maps = mapResponse.maps;
    if (!state.maps.length) { $('map-status').textContent = mapResponse.note || '未保留地图，仍可查看轨迹。'; return; }
    for (const [index, map] of state.maps.entries()) {
      const option = document.createElement('option'); option.value = index;
      option.textContent = map.label || `工作镜头 · 地图 ${map.index}`; $('map-select').append(option);
    }
    const preferred = state.maps.findIndex(map => map.coordinates === 'trajectory');
    $('map-select').value = Math.max(0, preferred);
    for (const name of ['map-select', 'show-map', 'show-keyframes', 'show-map-axes']) $(name).disabled = false;
    $('show-map').checked = true; scene.showMap = true;
    selectMap();
  } catch (error) {
    if (request === state.load) $('map-status').textContent = `地图未加载：${error.message}；轨迹仍可查看。`;
  }
}
function selectMap() {
  const map = state.maps[Number($('map-select').value)];
  if (!map) return;
  scene.setMap(map);
  const count = `${map.points.length.toLocaleString()} 点${map.point_count > map.points.length ? `（原始 ${map.point_count.toLocaleString()}）` : ''} · ${map.keyframes.length} 关键帧`;
  const alignment = map.alignment;
  const detail = map.alignment.source === 'fixed_scene' ? '固定场景坐标 · 验收残差不是真实误差' : map.coordinates === 'trajectory'
    ? `展示拟合 ${(alignment.rmse_m*1000).toFixed(1)} mm · 非真实误差`
    : `原始地图坐标 · 不叠加轨迹：${alignment.reason}`;
  $('map-status').textContent = `${count} · ${detail}`;
  updateMapLegend();
}
function updateMapLegend() {
  $('scene-scale').textContent = scene.rawMap ? '地图单位 · 朝向未对齐' : '单位 m · Z ↑';
  $('legend-track').parentElement.hidden = scene.rawMap;
  $('legend-current').hidden = scene.rawMap;
  $('legend-map').hidden = !scene.showMap || !scene.map;
  $('show-map-axes').disabled = !scene.showMap || !scene.map;
  $('show-world-axes').disabled = !state.recording || scene.rawMap;
  $('show-world-axes').title = scene.rawMap ? '此子地图未对齐，不能叠加世界坐标轴' : '轨迹世界原点与坐标轴，单位为米，Z 轴朝上';
}
function setTrack(track) {
  if (!state.recording?.tracks[track]) return;
  state.track = track; scene.track = track;
  $('legend-track').textContent = (state.recording.companion ? '左手 · 右手橙色 / ' : '') + {tcp: 'TCP 轨迹', camera: '相机轨迹', imu: 'IMU 轨迹'}[track];
  $('legend-color').style.background = state.recording.companion ? '#8ee3bf' : {tcp: '#8ee3bf', camera: '#91b7f4', imu: '#d9b67b'}[track];
  document.querySelectorAll('[data-track]').forEach(button => button.classList.toggle('active', button.dataset.track === track));
  $('stat-path').replaceChildren(document.createTextNode(state.recording.tracks[track].path_length.toFixed(2)), unit(' m'));
  $('stat-path').previousElementSibling.textContent = track === 'tcp' ? 'TCP 路程' : track === 'camera' ? '相机路程' : 'IMU 路程';
  const coverage = state.recording.summary.constraint_fraction?.full;
  $('stat-coverage').replaceChildren(document.createTextNode(Number.isFinite(coverage) ? (coverage * 100).toFixed(0) : '—'), unit(' %'));
  chart.load(state.recording.times, state.recording.tracks[track].speed); renderFrame(state.index);
  updateMapLegend();
}
function renderFrame(index, imageReady = false) {
  const recording = state.recording; if (!recording) return;
  index = clamp(Math.round(index), 0, recording.times.length - 1);
  if (state.frameMode && !imageReady && framePreview.displayed !== index) {
    framePreview.request(index, videoSeconds(recording.times[index]));
    return;
  }
  state.index = clamp(Math.round(index), 0, recording.times.length - 1);
  const track = recording.tracks[state.track], p = track.position[state.index], q = track.quaternion[state.index], frame = recording.frames[state.index];
  ['x', 'y', 'z'].forEach((axis, k) => { $(`pose-${axis}`).textContent = p[k].toFixed(4); });
  $('pose-speed').textContent = track.speed[state.index].toFixed(3);
  $('pose-constraint').textContent = labels[frame.constraint] || '未知'; $('pose-constraint').style.color = colors[frame.constraint] || colors.unknown;
  $('detail-quaternion').textContent = q.map(value => value.toFixed(3)).join('  ');
  const distanceReadout = tagDistanceReadout(recording, state.index);
  $('tag-distance-readout').hidden = !distanceReadout.isVisible;
  $('tag-distance-value').textContent = distanceReadout.text;
  $('timeline').value = state.index;
  $('frame-counter').textContent = `FRAME ${state.index + 1} / ${recording.times.length}`;
  $('video-frame-label').textContent = `SOURCE FRAME ${frame.video_frame}`;
  $('current-time').textContent = formatTime(recording.times[state.index] - recording.times[0]);
  scene.index = state.index; scene.draw(); chart.index = state.index; chart.draw();
  companionVideo.synchronize(recording.times[state.index], state.playing, Number($('rate').value));
}
function seek(index, stop = true) {
  if (!state.recording) return;
  if (stop) pause();
  index = clamp(Math.round(index), 0, state.recording.times.length - 1);
  state.playTime = state.recording.times[index]; renderFrame(index);
  if (state.videoReady) video.currentTime = videoSeconds(state.playTime);
}
function pause() { state.playing = false; video.pause(); companionVideo.synchronize(state.playTime, false); $('play').textContent = '▶'; $('play').setAttribute('aria-label', '播放'); }
async function play() {
  if (!state.recording) return;
  if (state.index >= state.recording.times.length - 1) seek(0);
  if (!state.frameMode) state.playTime = state.recording.times[state.index];
  state.lastTick = performance.now();
  state.playing = true; $('play').textContent = 'Ⅱ'; $('play').setAttribute('aria-label', '暂停');
  if (state.videoReady) {
    video.currentTime = videoSeconds(state.playTime);
    video.playbackRate = Number($('rate').value);
    await resumeVideo();
  }
}
async function resumeVideo() {
  const generation = state.load, source = state.videoSource;
  try { await video.play(); } catch (error) {
    if (generation === state.load && source === state.videoSource && state.playing && error.name !== 'AbortError') {
      if (!enableFrameMode()) pause();
    }
  }
}
function tick(now) {
  if (state.playing && state.recording) {
    const times = state.recording.times;
    if (state.videoReady) state.playTime = (state.recording.video_time_offset_s ?? times[0]) + video.currentTime;
    else state.playTime += Math.min((now - state.lastTick) / 1000, .2) * Number($('rate').value);
    if (state.playTime >= times.at(-1)) {
      if (state.loop) { state.playTime = times[0]; if (state.videoReady) { video.currentTime = videoSeconds(times[0]); video.play().catch(() => pause()); } }
      else { renderFrame(times.length - 1); pause(); }
    }
    const index = nearestIndex(times, state.playTime);
    if (index !== state.index) renderFrame(index);
  }
  state.lastTick = now; requestAnimationFrame(tick);
}
video.addEventListener('loadedmetadata', () => {
  if (state.frameMode || !state.videoSource || !video.currentSrc.endsWith(state.videoSource)) return;
  state.videoReady = true; $('video-empty').hidden = true;
  video.currentTime = videoSeconds(state.playTime);
  renderFrame(nearestIndex(state.recording.times, state.playTime));
  if (state.playing) resumeVideo();
});
function enableFrameMode() {
  if (state.frameMode) return true;
  if (!state.recording?.frame_url) {
    $('video-empty').hidden = false;
    $('video-empty').querySelector('p').textContent = '当前服务未提供兼容播放，请重启 umi-view 并刷新页面。';
    return false;
  }
  state.frameMode = true; state.videoReady = false; state.videoSource = null;
  video.pause(); video.removeAttribute('src'); video.load(); video.hidden = true;
  framePreview.reset(state.recording.frame_url);
  $('compatibility-mode').textContent = '兼容模式'; $('compatibility-mode').disabled = true;
  $('video-empty').hidden = false;
  $('video-empty').querySelector('p').textContent = '正在加载兼容画面…';
  $('video-note').textContent = '兼容播放 · 视频画面与轨迹同步';
  message('已启用视频兼容播放；画面按服务器可达帧率显示，轨迹与画面保持对应。');
  renderFrame(state.index);
  return true;
}
video.addEventListener('error', () => {
  if (!state.recording?.video_url || !video.getAttribute('src') || state.frameMode) return;
  enableFrameMode();
});
$('compatibility-mode').addEventListener('click', () => { state.manualFrameMode = true; enableFrameMode(); });
video.addEventListener('ended', () => { if (!state.recording) return; if (state.loop) { seek(0); play(); } else { seek(state.recording.times.length - 1); } });
$('play').addEventListener('click', () => state.playing ? pause() : play());
$('timeline').addEventListener('input', event => seek(Number(event.target.value)));
$('previous-frame').addEventListener('click', () => seek(state.index - 1));
$('next-frame').addEventListener('click', () => seek(state.index + 1));
$('rate').addEventListener('change', () => { video.playbackRate = Number($('rate').value); });
$('loop').addEventListener('click', () => { state.loop = !state.loop; $('loop').setAttribute('aria-pressed', String(state.loop)); });
$('show-pose').addEventListener('change', event => { scene.pose = event.target.checked; scene.draw(); });
$('show-overlay').addEventListener('change', event => { scene.overlay = event.target.checked; scene.draw(); });
$('map-select').addEventListener('change', selectMap);
$('show-map').addEventListener('change', event => {
  scene.showMap = event.target.checked; scene.fit(); scene.draw(); updateMapLegend();
});
$('show-keyframes').addEventListener('change', event => { scene.keyframes = event.target.checked; scene.draw(); });
$('show-map-axes').addEventListener('change', event => { scene.showMapAxes = event.target.checked; scene.draw(); });
$('show-world-axes').addEventListener('change', event => { scene.showWorldAxes = event.target.checked; scene.draw(); });
document.querySelectorAll('[data-track]').forEach(button => button.addEventListener('click', () => setTrack(button.dataset.track)));
function setView(view) { scene.view(view); document.querySelectorAll('[data-view]').forEach(button => button.classList.toggle('active', button.dataset.view === view)); }
document.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => setView(button.dataset.view)));
$('video-fullscreen').addEventListener('click', () => { const target = state.frameMode ? video.parentElement : video; if ((state.videoReady || state.frameMode) && target.requestFullscreen) target.requestFullscreen().catch(error => message(error.message, true)); });
$('help-button').addEventListener('click', () => $('help-dialog').showModal());
$('details-button').addEventListener('click', () => {
  if (!state.recording) return;
  $('details-text').textContent = JSON.stringify({directory: state.recording.path, calibration: state.recording.summary.calibration_used, extent_m: state.recording.tracks[state.track].extent, notes: state.recording.notes, summary: state.recording.summary}, null, 2);
  $('details-dialog').showModal();
});
document.querySelectorAll('[data-close]').forEach(button => button.addEventListener('click', () => $(button.dataset.close).close()));
$('refresh').addEventListener('click', () => refresh()); $('search').addEventListener('input', renderLibrary);
document.addEventListener('keydown', event => {
  if (/INPUT|SELECT|TEXTAREA/.test(event.target.tagName) || document.querySelector('dialog[open]')) return;
  if (event.code === 'Space') { event.preventDefault(); state.playing ? pause() : play(); }
  else if (event.key === 'ArrowLeft') { event.preventDefault(); seek(state.index - 1); }
  else if (event.key === 'ArrowRight') { event.preventDefault(); seek(state.index + 1); }
  else if (['r', 'R', '1', '2', '3', '4'].includes(event.key)) setView({'2': 'top', '3': 'front', '4': 'side'}[event.key] || 'home');
  else if (event.key === '?') $('help-dialog').showModal();
});
requestAnimationFrame(tick);
refresh(true);
