import test from 'node:test';
import assert from 'node:assert/strict';
import {nearestIndex, rotateQuaternion, formatTime, niceStep} from '../../software/viewer/static/math.mjs';

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

import {FramePreview} from '../../software/viewer/static/media.mjs';

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
