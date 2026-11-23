import {nearestIndex, formatTime} from './math.mjs';
import {TrajectoryScene} from './scene.mjs';
import {SpeedChart} from './chart.mjs';
import {createPlayback} from './playback.mjs';
import {api, createLibrary, recordingLabel} from './library.mjs';
import {formatTagDistance, tagDistanceReadout} from './measurements.mjs';
import {QualityTimeline, frameQuality, qualityLabels} from './quality.mjs';

const $ = id => document.getElementById(id);
const state = {recording: null, maps: [], track: 'tcp', load: 0};
const playback = createPlayback($, {renderFrame, message});
const library = createLibrary({
  search: $('search'), list: $('recording-list'), count: $('record-count'), root: $('root-path'),
  refreshButton: $('refresh'), onSelect: load, onError: text => message(text, true),
});
const seekTime = time => {
  if (state.recording) playback.seek(nearestIndex(state.recording.times, time));
};
const qualityTimeline = new QualityTimeline($('quality-timeline'), seekTime);
const scene = new TrajectoryScene($('trajectory-canvas'), index => playback.seek(index));
const chart = new SpeedChart($('speed-canvas'), seekTime);
const qualityColors = {visual: '#8ee3bf', natural: '#8ee3bf', map: '#8ee3bf',
  visual_and_poster: '#92d8d9', poster: '#91b7f4', rot: '#d9b67b', imu: '#e29a87', unknown: '#849597'};

function message(text = '', error = false) {
  $('message').textContent = text; $('message').className = `message${error ? ' error' : ''}`; $('message').hidden = !text;
}
function renderDualHandSummary(recording) {
  const root = $('dual-hand-summary');
  if (!recording.companion) { root.hidden = true; return; }
  const duration = recording.times.at(-1) - recording.times[0];
  const tracks = {
    [recording.primary_arm]: recording.tracks.tcp,
    [recording.companion.arm]: recording.companion_tracks?.tcp,
  };
  for (const arm of ['left', 'right']) {
    const card = root.querySelector(`[data-hand-summary="${arm}"]`);
    const track = tracks[arm];
    card.hidden = !track;
    if (!track) continue;
    card.querySelector('[data-hand-time]').textContent = `${duration.toFixed(2)} s`;
    card.querySelector('[data-hand-path]').textContent = `${track.path_length.toFixed(3)} m`;
  }
  root.hidden = false;
}
async function load(id) {
  const request = ++state.load;
  playback.pause(); message('正在加载轨迹和视频…'); $('play').disabled = true;
  try {
    const recording = await api(`/api/recordings/${id}/playback`);
    if (request !== state.load) return;
    state.recording = recording;
    state.track = 'tcp';
    const primaryLabel = recording.primary_arm === 'right' ? '右手' : '左手';
    $('video-title').textContent = recording.companion ? '双手视频' : '录像画面';
    $('video-hand-label').textContent = recording.primary_arm ? primaryLabel : '镜头';
    const paired = Boolean(recording.companion);
    $('inspector-grid').classList.toggle('paired-layout', paired);
    document.body.classList.toggle('paired-view', paired);
    $('primary-readout').hidden = !paired;
    $('video-fullscreen').textContent = paired ? '双路全屏' : '全屏';
    $('video-fullscreen').title = paired ? '全屏查看两路视频' : '全屏视频';
    qualityTimeline.load(recording);
    const summary = recording.summary, selection = summary.calibration_used || {};
    $('record-title').textContent = recordingLabel(recording.name); $('record-title').title = recording.name;
    $('record-meta').textContent = `${recording.name} · ${summary.scene?.frame_id || summary.mode || 'handheld'}`;
    $('status-badge').textContent = summary.mode === 'poster' ? '海报轨迹 · 任务坐标' : summary.mode === 'local' ? '独立回退 · 局部坐标' : summary.status === 'ok' ? '抽取成功' : summary.status === 'failed' ? '抽取失败' : '已加载轨迹';
    $('status-badge').classList.toggle('failed', summary.status === 'failed');
    $('stat-duration').replaceChildren(document.createTextNode((recording.times.at(-1) - recording.times[0]).toFixed(1)), unit(' s'));
    $('detail-serial').textContent = selection.serial || '未记录'; $('detail-calibration').textContent = selection.id || '未记录';
    if (recording.companion) {
      const other = recording.companion.summary.calibration_used || {};
      $('detail-serial').textContent = `L ${selection.serial} / R ${other.serial}`;
      $('detail-calibration').textContent = `L ${selection.id} / R ${other.id}`;
    }
    $('detail-rows').textContent = recording.times.length.toLocaleString();
    renderDualHandSummary(recording);
    $('timeline').max = recording.times.length - 1; $('timeline').disabled = false; $('play').disabled = false;
    $('total-time').textContent = formatTime(recording.times.at(-1) - recording.times[0]);
    $('scene-empty').hidden = true;
    for (const button of document.querySelectorAll('[data-track]')) { button.disabled = !recording.tracks[button.dataset.track]; button.classList.toggle('active', button.dataset.track === 'tcp'); }
    scene.load(recording);
    playback.load(recording);
    setTrack('tcp');
    library.setActive(recording.id);
    loadMaps(id, request);
    const url = new URL(location.href); url.searchParams.set('recording', id); history.replaceState(null, '', url);
    const warnings = [...recording.notes];
    if (summary.status === 'failed') warnings.unshift(`此结果被标记为失败：${summary.reason || '请查看本次记录'}。轨迹仅供排查。`);
    message(warnings.join(' '), summary.status === 'failed');
    document.body.dataset.loaded = 'true';
  } catch (error) {
    if (request === state.load) { message(`加载失败：${error.message}`, true); $('play').disabled = !state.recording; }
  }
}
function unit(text) { const element = document.createElement('small'); element.textContent = text; return element; }
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
  $('legend-companion').hidden = scene.rawMap || !state.recording?.companion_tracks?.[state.track];
  $('legend-map').hidden = !scene.showMap || !scene.map;
  $('show-map-axes').disabled = !scene.showMap || !scene.map;
  $('show-world-axes').disabled = !state.recording || scene.rawMap;
  $('show-world-axes').title = scene.rawMap ? '此子地图未对齐，不能叠加世界坐标轴' : '轨迹世界原点与坐标轴，单位为米，Z 轴朝上';
}
function setTrack(track) {
  if (!state.recording?.tracks[track]) return;
  state.track = track; scene.track = track;
  const primaryLabel = state.recording.primary_arm === 'right' ? '右手' : '左手';
  const companionLabel = state.recording.companion?.arm === 'right' ? '右手' : '左手';
  const paired = Boolean(state.recording.companion);
  const referenceNames = {tcp: 'TCP', camera: '相机光心', imu: 'IMU 原点'};
  const reference = referenceNames[track];
  $('legend-track').textContent = `${paired ? primaryLabel + ' ' : ''}${reference} 轨迹`;
  $('legend-companion-text').textContent = `${companionLabel} ${reference} 轨迹`;
  const overlayNames = Object.keys(state.recording.tracks).filter(name => name !== track)
    .map(name => referenceNames[name]);
  $('overlay-label').textContent = paired ? `叠加${primaryLabel}其他参考点` : '叠加其他参考点';
  $('overlay-control').title = overlayNames.length
    ? `用虚线叠加${paired ? primaryLabel + '的' : '同一设备的'}${overlayNames.join('、')}的轨迹`
    : '本次记录没有其他参考点轨迹';
  const chartLegend = $('chart-hand-legend');
  chartLegend.hidden = !state.recording.companion_tracks?.[track];
  chartLegend.querySelector('[data-primary]').textContent = primaryLabel;
  chartLegend.querySelector('[data-companion]').textContent = companionLabel;
  $('legend-color').style.background = state.recording.companion ? '#8ee3bf' : {tcp: '#8ee3bf', camera: '#91b7f4', imu: '#d9b67b'}[track];
  document.querySelectorAll('[data-track]').forEach(button => button.classList.toggle('active', button.dataset.track === track));
  $('stat-path').replaceChildren(document.createTextNode(state.recording.tracks[track].path_length.toFixed(2)), unit(' m'));
  $('stat-path').previousElementSibling.textContent = track === 'tcp' ? 'TCP 路程' : track === 'camera' ? '相机路程' : 'IMU 路程';
  const summary = state.recording.summary;
  const coverage = summary.constraint_fraction?.full;
  $('coverage-label').textContent = summary.reconstruction_backend === 'poster_vio' ? '自然特征覆盖'
    : summary.mode === 'scene' ? '地图定位覆盖' : '视觉定位覆盖';
  $('stat-coverage').replaceChildren(document.createTextNode(Number.isFinite(coverage) ? (coverage * 100).toFixed(0) : '—'), unit(' %'));
  chart.load(state.recording.times, state.recording.tracks[track].speed,
    state.recording.companion_tracks?.[track]?.speed || null); playback.redraw();
  updateMapLegend();
}
function frameLabel(frame) {
  if (frame?.source_video_frame != null) return `原片帧 ${frame.source_video_frame}`;
  return frame?.video_frame != null ? `视频帧 ${frame.video_frame}` : '帧 —';
}
function renderHandReadout(root, hand, index) {
  root.querySelector('[data-quality]').textContent = qualityLabels[frameQuality(hand.frames?.[index], hand.summary)];
  const distance = root.querySelector('[data-distance]');
  distance.hidden = !hand.tag_distance;
  if (!distance.hidden) {
    const value = formatTagDistance(hand.tag_distance, index);
    distance.textContent = `标签中心距 ${value}${value === '—' ? '' : ' mm'}`;
  }
}
function renderFrame(index) {
  const recording = state.recording;
  const track = recording.tracks[state.track], p = track.position[index], q = track.quaternion[index], frame = recording.frames[index];
  ['x', 'y', 'z'].forEach((axis, k) => { $(`pose-${axis}`).textContent = p[k].toFixed(4); });
  $('pose-speed').textContent = track.speed[index].toFixed(3);
  const quality = frameQuality(frame, recording.summary);
  $('pose-constraint').textContent = qualityLabels[quality];
  $('pose-constraint').style.color = qualityColors[quality];
  $('detail-quaternion').textContent = q.map(value => value.toFixed(3)).join('  ');
  const time = recording.times[index];
  if (recording.companion) {
    const otherIndex = nearestIndex(recording.companion.times, time);
    renderHandReadout($('primary-readout'), recording, index);
    const otherRoot = $('companion-video');
    renderHandReadout(otherRoot, recording.companion, otherIndex);
    otherRoot.querySelector('[data-frame-label]').textContent = frameLabel(recording.companion.frames?.[otherIndex]);
    $('tag-distance-readout').hidden = true;
  } else {
    const distanceReadout = tagDistanceReadout(recording, index);
    $('tag-distance-readout').hidden = !distanceReadout.isVisible;
    $('tag-distance-value').textContent = distanceReadout.text;
  }
  $('timeline').value = index;
  $('frame-counter').textContent = `FRAME ${index + 1} / ${recording.times.length}`;
  $('video-frame-label').textContent = frameLabel(frame);
  $('current-time').textContent = formatTime(recording.times[index] - recording.times[0]);
  scene.index = index; scene.draw(); chart.index = index; chart.draw();
  qualityTimeline.select(time);
}
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
$('help-button').addEventListener('click', () => $('help-dialog').showModal());
$('details-button').addEventListener('click', async () => {
  const recording = state.recording;
  if (!recording) return;
  const track = state.track;
  const dialog = $('details-dialog');
  const content = $('details-text');
  content.textContent = '正在读取本次记录…';
  dialog.showModal();
  try {
    const details = await api(`/api/recordings/${recording.id}/details`);
    if (state.recording !== recording || !dialog.open) return;
    content.textContent = JSON.stringify({...details, extent_m: recording.tracks[track].extent, notes: recording.notes}, null, 2);
  } catch (error) {
    if (state.recording === recording && dialog.open) content.textContent = `读取失败：${error.message}`;
  }
});
document.querySelectorAll('[data-close]').forEach(button => button.addEventListener('click', () => $(button.dataset.close).close()));
document.addEventListener('keydown', event => {
  if (/INPUT|SELECT|TEXTAREA/.test(event.target.tagName) || document.querySelector('dialog[open]')) return;
  if (event.code === 'Space' && event.target.tagName !== 'BUTTON') { event.preventDefault(); playback.toggle(); }
  else if (event.key === 'ArrowLeft') { event.preventDefault(); playback.step(-1); }
  else if (event.key === 'ArrowRight') { event.preventDefault(); playback.step(1); }
  else if (['r', 'R', '1', '2', '3', '4'].includes(event.key)) setView({'2': 'top', '3': 'front', '4': 'side'}[event.key] || 'home');
  else if (event.key === '?') $('help-dialog').showModal();
});
library.refresh(true);
