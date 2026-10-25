import test from 'node:test';
import assert from 'node:assert/strict';
import {nearestIndex, rotateQuaternion, formatTime, niceStep} from '../../software/viewer/static/math.mjs';
import {TrajectoryScene} from '../../software/viewer/static/scene.mjs';
import {CompanionVideo} from '../../software/viewer/static/paired.mjs';

test('video time maps to nearest trajectory frame including boundaries', () => {
  assert.equal(nearestIndex([0, .1, .2, .3], .18), 2);
  assert.equal(nearestIndex([0, .1, .2, .3], -1), 0);
  assert.equal(nearestIndex([0, .1, .2, .3], 1), 3);
});
test('xyzw quaternion rotates local pose axes into world coordinates', () => {
  const result = rotateQuaternion([1, 0, 0], [0, 0, Math.SQRT1_2, Math.SQRT1_2]);
  assert.ok(Math.abs(result[0]) < 1e-12);
  assert.ok(Math.abs(result[1] - 1) < 1e-12);
  assert.deepEqual(rotateQuaternion([1, 2, 3], [0, 0, 0, 1]), [1, 2, 3]);
});
test('time labels and grid spacing are stable', () => {
  assert.equal(formatTime(34.701), '00:34.70');
  assert.equal(formatTime(61.25), '01:01.25');
  assert.equal(niceStep(.094), .1);
});

test('raw maps keep their own bounds and never mix in the metric trajectory', () => {
  const scene = Object.create(TrajectoryScene.prototype);
  scene.data = {times: [0, .5, 1], tracks: {tcp: {position: [[1000, 1000, 1000], [1001, 1001, 1001]]}}};
  scene.track = 'tcp'; scene.showMap = true;
  scene.map = {coordinates: 'map', bounds: [[0, 0, 0], [2, 2, 2]], keyframes: []};
  scene.fit();
  assert.equal(scene.rawMap, true);
  assert.deepEqual(scene.center, [1, 1, 1]);
  scene.showMap = false; scene.fit();
  assert.equal(scene.rawMap, false);
  assert.deepEqual(scene.center, [1000.5, 1000.5, 1000.5]);
});

test('clicking a keyframe in an unaligned map seeks its corresponding video time', () => {
  const scene = Object.create(TrajectoryScene.prototype);
  scene.canvas = {getBoundingClientRect: () => ({left: 0, top: 0})};
  scene.project = point => point;
  scene.data = {times: [0, .5, 1]}; scene.showMap = true;
  scene.map = {coordinates: 'map', keyframes: [{position: [20, 30, 0], time: .52}]};
  let picked = -1; scene.onPick = index => { picked = index; };
  scene.pick({clientX: 20, clientY: 30});
  assert.equal(picked, 1);
});

test('map axes use the same similarity transform as aligned points', () => {
  const scene = Object.create(TrajectoryScene.prototype);
  scene.span = 10; scene.showMap = true;
  scene.map = {coordinates: 'trajectory', alignment: {scale: 2,
    rotation: [[0, -1, 0], [1, 0, 0], [0, 0, 1]], translation: [3, 4, 5]}};
  const frame = scene.mapAxes();
  assert.deepEqual(frame.origin, [3, 4, 5]);
  assert.equal(frame.length, 1);
  assert.deepEqual(frame.axes.map(axis => axis.end), [[3, 6, 5], [1, 4, 5], [3, 4, 7]]);
  scene.map = {coordinates: 'map'};
  const raw = scene.mapAxes();
  assert.deepEqual(raw.origin, [0, 0, 0]);
  assert.deepEqual(raw.axes.map(axis => axis.end), [[2, 0, 0], [0, 2, 0], [0, 0, 2]]);
  scene.showMap = false;
  assert.equal(scene.mapAxes(), null);
});

test('map origin stays in view when axes are enabled', () => {
  const scene = Object.create(TrajectoryScene.prototype);
  scene.data = {tracks: {tcp: {position: [[100, 100, 100], [102, 102, 102]]}}};
  scene.track = 'tcp'; scene.showMap = true; scene.showMapAxes = true; scene.span = 1;
  scene.map = {coordinates: 'map', bounds: [[100, 100, 100], [102, 102, 102]], keyframes: []};
  scene.fit();
  assert.deepEqual(scene.center, [51, 51, 51]);
  scene.showMapAxes = false; scene.fit();
  assert.deepEqual(scene.center, [101, 101, 101]);
});

test('world axes stay at the metric origin while map axes follow the alignment', () => {
  const scene = Object.create(TrajectoryScene.prototype);
  scene.data = {}; scene.span = 10; scene.showMap = true;
  scene.map = {index: 2, coordinates: 'trajectory', alignment: {scale: 2,
    rotation: [[0, -1, 0], [1, 0, 0], [0, 0, 1]], translation: [3, 4, 5]}};
  const world = scene.worldAxes(), map = scene.mapAxes();
  assert.deepEqual(world.origin, [0, 0, 0]);
  assert.deepEqual(map.origin, [3, 4, 5]);
  assert.equal(world.unit, 'm');
  assert.equal(map.unit, '地图单位');
  assert.deepEqual(world.axes.map(axis => axis.end), [[world.length, 0, 0], [0, world.length, 0], [0, 0, world.length]]);
  assert.equal(map.originLabel, 'O_map2');
  scene.map = {coordinates: 'map'};
  assert.equal(scene.worldAxes(), null);
  scene.showMap = false;
  assert.deepEqual(scene.worldAxes().origin, [0, 0, 0]);
});

test('fitting both frames includes both origins without mixing their coordinates', () => {
  const scene = Object.create(TrajectoryScene.prototype);
  scene.data = {tracks: {tcp: {position: [[100, 100, 100], [102, 102, 102]]}}};
  scene.track = 'tcp'; scene.span = 1; scene.showMap = true;
  scene.showMapAxes = true; scene.showWorldAxes = true;
  scene.map = {coordinates: 'trajectory', bounds: [[100, 100, 100], [102, 102, 102]], keyframes: [],
    alignment: {scale: 1, rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], translation: [104, 104, 104]}};
  scene.fit();
  for (const frame of [scene.worldAxes(), scene.mapAxes()]) {
    for (const point of [frame.origin, ...frame.axes.map(axis => axis.end)]) {
      assert.ok(point.every((value, k) => Math.abs(value - scene.center[k]) <= scene.span / 2));
    }
  }
  scene.showWorldAxes = false; scene.fit();
  assert.ok(scene.center.every(value => value > 100));
  scene.showMapAxes = false; scene.fit();
  assert.deepEqual(scene.center, [101, 101, 101]);
});

import {FramePreview, VideoPreview} from '../../software/viewer/static/media.mjs';

test('ready browser preview is loaded without polling again', async () => {
  const urls = [];
  const preview = new VideoPreview(url => urls.push(url), error => { throw error; }, {
    fetcher: async () => ({ok: true, json: async () => ({status: 'ready', url: '/preview.webm'})}),
  });
  await preview.reset('/status');
  assert.deepEqual(urls, ['/preview.webm']);
  assert.equal(preview.url, null);
  preview.reset();
});

test('switching recordings discards an old preview response', async () => {
  let finishOld; const urls = [];
  const preview = new VideoPreview(url => urls.push(url), error => { throw error; }, {
    fetcher: url => url === '/old' ? new Promise(resolve => { finishOld = resolve; })
      : Promise.resolve({ok: true, json: async () => ({status: 'ready', url: '/new.webm'})}),
  });
  const old = preview.reset('/old');
  await preview.reset('/new');
  finishOld({ok: true, json: async () => ({status: 'ready', url: '/old.webm'})});
  await old;
  assert.deepEqual(urls, ['/new.webm']);
  preview.reset();
});

function imageStub() { return {hidden: true, removeAttribute() { this.src = ''; }}; }
function jpegResponse() { return {ok: true, blob: async () => new Blob(['jpeg'], {type: 'image/jpeg'})}; }

test('compatibility frames coalesce pending requests and present matching indices', async () => {
  let resolveFirst; const frames = []; const requests = [];
  const preview = new FramePreview(imageStub(), index => frames.push(index), error => { throw error; }, {
    fetcher: url => { requests.push(url); return requests.length === 1 ? new Promise(resolve => { resolveFirst = resolve; }) : Promise.resolve(jpegResponse()); },
    decode: async () => {},
  });
  preview.reset('/frame');
  preview.request(1, .1); clearTimeout(preview.timer);
  const first = preview.pump();
  preview.request(2, .2); preview.request(3, .3);
  resolveFirst(jpegResponse()); await first; clearTimeout(preview.timer);
  await preview.pump();
  assert.deepEqual(frames, [1, 3]);
  assert.deepEqual(requests, ['/frame?t=0.100000', '/frame?t=0.300000']);
  assert.equal(preview.image.hidden, false);
  preview.reset();
});

test('switching recordings prevents an old image from overwriting the new recording', async () => {
  let resolveFetch; const frames = [];
  const preview = new FramePreview(imageStub(), i => frames.push(i), error => { throw error; }, {
    fetcher: () => new Promise(resolve => { resolveFetch = resolve; }), decode: async () => {},
  });
  preview.reset('/old'); preview.request(20, 2); clearTimeout(preview.timer);
  const old = preview.pump(); preview.reset('/new');
  resolveFetch(jpegResponse()); await old;
  assert.deepEqual(frames, []); assert.equal(preview.image.hidden, true);
  preview.reset();
});

test('decode failures surface a useful message and stop retrying', async () => {
  let message = '';
  const preview = new FramePreview(imageStub(), () => {}, error => { message = error.message; }, {
    fetcher: async () => ({ok: false, json: async () => ({error: 'FFmpeg 不可用'})}), decode: async () => {},
  });
  preview.reset('/frame'); preview.request(0, 0); clearTimeout(preview.timer); await preview.pump();
  assert.equal(message, 'FFmpeg 不可用'); assert.equal(preview.url, null);
  preview.reset();
});

test('companion playback keeps the explicit offset after trimming to common episode time', async () => {
  const image = imageStub(), note = {}, arm = {}, attributes = new Map();
  const video = {readyState: 2, currentTime: 0, paused: true, seeking: false,
    pause() { this.paused = true; }, play() { this.paused = false; return Promise.resolve(); }, load() {},
    addEventListener() {}, removeAttribute(name) { attributes.delete(name); }, getAttribute(name) { return attributes.get(name); },
    canPlayType() { return 'probably'; }, set src(value) { attributes.set('src', value); }};
  const root = {querySelector(selector) { return {'video': video, 'img': image, '[data-note]': note, '[data-arm]': arm}[selector]; }};
  const preview = new CompanionVideo(root);
  preview.reset({arm: 'right', times: [.2, .3, .4], video_time_offset_s: .15, video_url: '/right.mp4', frame_url: '/right/frame'});
  preview.synchronize(.35, false);
  assert.ok(Math.abs(video.currentTime-.2) < 1e-10);
  preview.synchronize(.4, true, 2);
  await Promise.resolve();
  assert.equal(video.paused, false); assert.equal(video.playbackRate, 2);
  const requests = [];
  preview.frames.request = (index, seconds) => requests.push({index, seconds});
  preview.fallback(); preview.synchronize(.31, false);
  assert.equal(requests.at(-1).index, 1);
  assert.ok(Math.abs(requests.at(-1).seconds-.15) < 1e-10);
  preview.reset(); assert.equal(root.hidden, true); assert.equal(video.paused, true);
});
