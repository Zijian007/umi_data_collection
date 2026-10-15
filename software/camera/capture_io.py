"""Read/write and validate the canonical capture directory defined in ``contracts.py``.

Dependency budget on the READ path: numpy + the Python standard library.  No OpenCV, no PyAV, no
camera SDK.  That is what lets the algorithm layer consume a capture in a process where
``software.camera.x3`` is not importable at all.

The image codec below is a deliberately small, pure-``zlib`` 8-bit grayscale PNG reader/writer.
The writer always uses filter type 0 so its output is byte-deterministic for a given zlib build;
the reader supports all five PNG filter types so it can also read PNGs written by other tools
(OpenCV, ImageMagick, ...).  Only colour type 0 (grayscale), bit depth 8, non-interlaced is
supported -- anything else raises, rather than silently converting.
"""

from __future__ import annotations

import csv
import io
import json
import struct
import zlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from software.camera.contracts import (
    CameraModel,
    CaptureManifest,
    ClockSpec,
    ContractError,
    FileRef,
    FrameSample,
    FrameStreamSpec,
    ImuSample,
    ImuStreamSpec,
    Producer,
    SCHEMA_VERSION,
    STANDARD_GRAVITY_M_S2,
    sha256_bytes,
    sha256_file,
    validate_manifest,
)

FRAMES_DIR = "frames"
IMU_DIR = "imu"
MANIFEST_NAME = "manifest.json"
CAMERA_MODEL_NAME = "camera_model.json"
FRAME_INDEX_NAME = "frames/index.csv"
FRAME_HASHES_NAME = "frames/images.sha256"
IMU_DATA_NAME = "imu/imu.csv"

FRAME_INDEX_HEADER = "#timestamp_ns,frame_index_raw,image_ref"
IMU_HEADER = "#timestamp_ns,gyro_x_rad_s,gyro_y_rad_s,gyro_z_rad_s,accel_x_m_s2,accel_y_m_s2,accel_z_m_s2"

#: ``%.17g`` round-trips every IEEE-754 double exactly, so the canonical CSV loses nothing.
_FLOAT_FMT = "%.17g"

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# minimal pure-stdlib grayscale PNG codec
# ---------------------------------------------------------------------------


def _chunk(typ: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + typ + data + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)


def encode_gray8_png(arr: np.ndarray, compress_level: int = 6) -> bytes:
    """Encode a 2-D uint8 array as a grayscale PNG (colour type 0, filter 0, non-interlaced)."""
    a = np.ascontiguousarray(arr)
    if a.ndim != 2 or a.dtype != np.uint8:
        raise ContractError(f"encode_gray8_png expects a 2-D uint8 array, got {a.shape} {a.dtype}")
    h, w = a.shape
    raw = np.zeros((h, w + 1), dtype=np.uint8)
    raw[:, 1:] = a                      # filter byte 0 in column 0
    out = bytearray(_PNG_MAGIC)
    out += _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
    out += _chunk(b"IDAT", zlib.compress(raw.tobytes(), compress_level))
    out += _chunk(b"IEND", b"")
    return bytes(out)


def _png_chunks(data: bytes) -> Iterator[tuple[bytes, bytes]]:
    if not data.startswith(_PNG_MAGIC):
        raise ContractError("not a PNG file (bad magic)")
    pos = len(_PNG_MAGIC)
    while pos + 8 <= len(data):
        (n,) = struct.unpack(">I", data[pos:pos + 4])
        typ = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + n]
        (crc,) = struct.unpack(">I", data[pos + 8 + n:pos + 12 + n])
        if zlib.crc32(typ + body) & 0xFFFFFFFF != crc:
            raise ContractError(f"PNG chunk {typ!r} failed CRC")
        yield typ, body
        pos += 12 + n
        if typ == b"IEND":
            return
    raise ContractError("PNG ended without IEND")


def png_gray8_size(path: Path) -> tuple[int, int]:
    """Read only the IHDR of a PNG and return ``(width, height)``.  Cheap dimension check."""
    with open(path, "rb") as f:
        head = f.read(33)
    if not head.startswith(_PNG_MAGIC):
        raise ContractError(f"{path}: not a PNG file")
    w, h, depth, ctype, _comp, _filt, interlace = struct.unpack(">IIBBBBB", head[16:29])
    if depth != 8 or ctype != 0 or interlace != 0:
        raise ContractError(f"{path}: expected 8-bit grayscale non-interlaced PNG, got "
                            f"depth={depth} colour_type={ctype} interlace={interlace}")
    return int(w), int(h)


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def decode_gray8_png(data: bytes) -> np.ndarray:
    """Decode an 8-bit grayscale non-interlaced PNG into a 2-D uint8 array (stdlib + numpy only)."""
    w = h = None
    idat = bytearray()
    for typ, body in _png_chunks(data):
        if typ == b"IHDR":
            w, h, depth, ctype, comp, filt, interlace = struct.unpack(">IIBBBBB", body)
            if depth != 8 or ctype != 0 or comp != 0 or filt != 0 or interlace != 0:
                raise ContractError(f"unsupported PNG: depth={depth} colour_type={ctype} comp={comp} "
                                    f"filter={filt} interlace={interlace}")
        elif typ == b"IDAT":
            idat += body
    if w is None:
        raise ContractError("PNG without IHDR")
    raw = zlib.decompress(bytes(idat))
    if len(raw) != h * (w + 1):
        raise ContractError(f"PNG payload is {len(raw)} bytes, expected {h * (w + 1)}")
    out = np.empty((h, w), dtype=np.uint8)
    prev = np.zeros(w, dtype=np.uint8)
    pos = 0
    for y in range(h):
        ft = raw[pos]
        pos += 1
        line = np.frombuffer(raw, dtype=np.uint8, count=w, offset=pos).astype(np.uint16)
        pos += w
        if ft == 0:
            cur = line
        elif ft == 1:                                   # Sub
            cur = line.copy()
            for x in range(1, w):
                cur[x] = (cur[x] + cur[x - 1]) & 0xFF
        elif ft == 2:                                   # Up
            cur = (line + prev.astype(np.uint16)) & 0xFF
        elif ft == 3:                                   # Average
            cur = line.copy()
            up = prev.astype(np.uint16)
            for x in range(w):
                left = int(cur[x - 1]) if x else 0
                cur[x] = (int(cur[x]) + ((left + int(up[x])) >> 1)) & 0xFF
        elif ft == 4:                                   # Paeth
            cur = line.copy()
            up = prev.astype(np.uint16)
            for x in range(w):
                left = int(cur[x - 1]) if x else 0
                upleft = int(up[x - 1]) if x else 0
                cur[x] = (int(cur[x]) + _paeth(left, int(up[x]), upleft)) & 0xFF
        else:
            raise ContractError(f"unknown PNG filter type {ft} on row {y}")
        row = cur.astype(np.uint8)
        out[y] = row
        prev = row
    return out


def read_gray8_png(path: Path) -> np.ndarray:
    return decode_gray8_png(Path(path).read_bytes())


# ---------------------------------------------------------------------------
# in-memory capture
# ---------------------------------------------------------------------------


@dataclass
class ImuStream:
    """Columnar IMU stream.  ``samples()`` yields the per-sample :class:`ImuSample` records."""

    timestamp_ns: np.ndarray       # int64, shape (N,)
    gyro_rad_s: np.ndarray         # float64, shape (N, 3)
    accel_m_s2: np.ndarray         # float64, shape (N, 3)
    frame_id: str
    range_source: str

    def __post_init__(self):
        self.timestamp_ns = np.asarray(self.timestamp_ns, dtype=np.int64).reshape(-1)
        self.gyro_rad_s = np.asarray(self.gyro_rad_s, dtype=np.float64).reshape(-1, 3)
        self.accel_m_s2 = np.asarray(self.accel_m_s2, dtype=np.float64).reshape(-1, 3)
        n = len(self.timestamp_ns)
        if len(self.gyro_rad_s) != n or len(self.accel_m_s2) != n:
            raise ContractError("ImuStream: timestamp/gyro/accel length mismatch")

    def __len__(self) -> int:
        return int(len(self.timestamp_ns))

    def samples(self) -> Iterator[ImuSample]:
        for i in range(len(self)):
            yield ImuSample(timestamp_ns=int(self.timestamp_ns[i]),
                            gyro_rad_s=tuple(float(x) for x in self.gyro_rad_s[i]),
                            accel_m_s2=tuple(float(x) for x in self.accel_m_s2[i]),
                            frame_id=self.frame_id, range_source=self.range_source)


@dataclass
class CanonicalCapture:
    """A canonical capture loaded from disk."""

    root: Path
    manifest: CaptureManifest
    camera_model: CameraModel
    frames: list[FrameSample]
    imu: ImuStream

    def image(self, i: int) -> np.ndarray:
        return read_gray8_png(self.root / self.frames[i].image_ref)

    def images(self) -> Iterator[np.ndarray]:
        for i in range(len(self.frames)):
            yield self.image(i)


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _file_ref(root: Path, rel: str, role: str) -> FileRef:
    p = root / rel
    return FileRef(path=rel, bytes=p.stat().st_size, sha256=sha256_file(p), role=role)


def write_capture(
    out_dir: Path,
    *,
    capture_id: str,
    camera_id: str,
    camera_profile_id: str,
    camera_profile_status: str,
    camera_model: CameraModel,
    clock: ClockSpec,
    producer: Producer,
    frames_meta: Iterable[tuple[int, int]],
    images: Iterable[np.ndarray],
    imu: ImuStream,
    source_files: Iterable[FileRef] = (),
    frame_step_raw: int = 1,
    exposure_convention: str = "unknown",
    imu_axes_note: str = "",
    imu_scaling: dict | None = None,
    imu_integrity: dict | None = None,
    gravity_m_s2: float = STANDARD_GRAVITY_M_S2,
    notes: Iterable[str] = (),
    png_compress_level: int = 6,
) -> CaptureManifest:
    """Write a canonical capture directory and return the manifest that was written.

    ``frames_meta`` is an iterable of ``(timestamp_ns, frame_index_raw)`` aligned with ``images``.
    Image file names are ``frames/<frame_index_raw:06d>.png`` -- derived from the RAW frame index,
    not from the timestamp, so the file name never depends on a clock convention.
    """
    root = Path(out_dir)
    (root / FRAMES_DIR).mkdir(parents=True, exist_ok=True)
    (root / IMU_DIR).mkdir(parents=True, exist_ok=True)

    index_rows: list[str] = [FRAME_INDEX_HEADER]
    hash_rows: list[str] = []
    n_frames = 0
    prev_ts: int | None = None
    for (ts_ns, idx_raw), img in zip(frames_meta, images, strict=True):
        ts_ns, idx_raw = int(ts_ns), int(idx_raw)
        if prev_ts is not None and ts_ns <= prev_ts:
            raise ContractError(f"frame timestamps must strictly increase: {prev_ts} -> {ts_ns}")
        prev_ts = ts_ns
        rel = f"{FRAMES_DIR}/{idx_raw:06d}.png"
        blob = encode_gray8_png(img, png_compress_level)
        (root / rel).write_bytes(blob)
        index_rows.append(f"{ts_ns},{idx_raw},{rel}")
        hash_rows.append(f"{sha256_bytes(blob)}  {rel}")
        n_frames += 1
    _write_text(root / FRAME_INDEX_NAME, "\n".join(index_rows) + "\n")
    _write_text(root / FRAME_HASHES_NAME, "\n".join(hash_rows) + ("\n" if hash_rows else ""))

    imu_rows = [IMU_HEADER]
    for i in range(len(imu)):
        g = imu.gyro_rad_s[i]
        a = imu.accel_m_s2[i]
        imu_rows.append(
            "%d,%s,%s,%s,%s,%s,%s" % (
                int(imu.timestamp_ns[i]),
                _FLOAT_FMT % g[0], _FLOAT_FMT % g[1], _FLOAT_FMT % g[2],
                _FLOAT_FMT % a[0], _FLOAT_FMT % a[1], _FLOAT_FMT % a[2]))
    _write_text(root / IMU_DATA_NAME, "\n".join(imu_rows) + "\n")

    _write_text(root / CAMERA_MODEL_NAME,
                json.dumps(camera_model.to_dict(), indent=1, ensure_ascii=False) + "\n")

    frames_spec = FrameStreamSpec(
        index_ref=FRAME_INDEX_NAME, image_dir=FRAMES_DIR, image_format="png_gray8",
        image_sha256_ref=FRAME_HASHES_NAME, n=n_frames, frame_step_raw=int(frame_step_raw),
        constant_fields={"camera_id": camera_id, "camera_model_id": camera_model.model_id,
                         "profile_id": camera_profile_id, "exposure_convention": exposure_convention})
    imu_spec = ImuStreamSpec(
        data_ref=IMU_DATA_NAME, n=len(imu), gravity_m_s2=float(gravity_m_s2),
        axes_note=imu_axes_note,
        constant_fields={"frame_id": imu.frame_id, "range_source": imu.range_source},
        scaling=dict(imu_scaling or {}), integrity=dict(imu_integrity or {}))

    ts = np.asarray(imu.timestamp_ns, dtype=np.int64)
    counts = {
        "n_frames": n_frames,
        "n_imu": len(imu),
        "frame_t_first_ns": int(index_rows[1].split(",")[0]) if n_frames else None,
        "frame_t_last_ns": int(index_rows[-1].split(",")[0]) if n_frames else None,
        "imu_t_first_ns": int(ts[0]) if len(imu) else None,
        "imu_t_last_ns": int(ts[-1]) if len(imu) else None,
    }

    outputs = [
        _file_ref(root, CAMERA_MODEL_NAME, "camera_model"),
        _file_ref(root, FRAME_INDEX_NAME, "frame_index"),
        _file_ref(root, FRAME_HASHES_NAME, "frame_image_hashes"),
        _file_ref(root, IMU_DATA_NAME, "imu"),
    ]
    manifest = CaptureManifest(
        schema_version=SCHEMA_VERSION, capture_id=capture_id, camera_id=camera_id,
        camera_profile_id=camera_profile_id, camera_profile_status=camera_profile_status,
        camera_model_id=camera_model.model_id, camera_model_ref=CAMERA_MODEL_NAME,
        clock=clock, frames=frames_spec, imu=imu_spec,
        source_files=list(source_files), outputs=outputs, producer=producer,
        counts=counts, notes=list(notes))
    problems = validate_manifest(manifest, camera_model)
    if problems:
        raise ContractError("refusing to write an invalid manifest: " + "; ".join(problems))
    _write_text(root / MANIFEST_NAME, manifest.to_json() + "\n")
    return manifest


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


def _read_rows(path: Path) -> list[list[str]]:
    with io.open(path, "r", encoding="utf-8", newline="") as f:
        return [r for r in csv.reader(f) if r and not r[0].lstrip().startswith("#")]


def read_manifest(root: Path) -> CaptureManifest:
    return CaptureManifest.from_json((Path(root) / MANIFEST_NAME).read_text(encoding="utf-8"))


def read_camera_model(root: Path, manifest: CaptureManifest | None = None) -> CameraModel:
    ref = (manifest.camera_model_ref if manifest is not None else CAMERA_MODEL_NAME)
    return CameraModel.from_dict(json.loads((Path(root) / ref).read_text(encoding="utf-8")))


def read_frames(root: Path, manifest: CaptureManifest) -> list[FrameSample]:
    const = manifest.frames.constant_fields
    rows = _read_rows(Path(root) / manifest.frames.index_ref)
    out = []
    for r in rows:
        if len(r) != 3:
            raise ContractError(f"{manifest.frames.index_ref}: expected 3 columns, got {r!r}")
        out.append(FrameSample(
            timestamp_ns=int(r[0]), frame_index_raw=int(r[1]), image_ref=r[2],
            camera_id=const.get("camera_id", manifest.camera_id),
            camera_model_id=const.get("camera_model_id", manifest.camera_model_id),
            profile_id=const.get("profile_id", manifest.camera_profile_id),
            exposure_convention=const.get("exposure_convention", "unknown"),
            schema_version=manifest.schema_version))
    return out


def read_imu(root: Path, manifest: CaptureManifest) -> ImuStream:
    rows = _read_rows(Path(root) / manifest.imu.data_ref)
    n = len(rows)
    ts = np.empty(n, dtype=np.int64)
    gyro = np.empty((n, 3), dtype=np.float64)
    acc = np.empty((n, 3), dtype=np.float64)
    for i, r in enumerate(rows):
        if len(r) != 7:
            raise ContractError(f"{manifest.imu.data_ref}: expected 7 columns, got {r!r}")
        ts[i] = int(r[0])
        gyro[i] = [float(r[1]), float(r[2]), float(r[3])]
        acc[i] = [float(r[4]), float(r[5]), float(r[6])]
    const = manifest.imu.constant_fields
    return ImuStream(ts, gyro, acc, frame_id=const.get("frame_id", "unknown"),
                     range_source=const.get("range_source", "unknown"))


def read_capture(root: Path) -> CanonicalCapture:
    """Load a canonical capture.  Raises :class:`ContractError` on a schema violation."""
    root = Path(root)
    manifest = read_manifest(root)
    model = read_camera_model(root, manifest)
    problems = validate_manifest(manifest, model)
    if problems:
        raise ContractError(f"{root}: invalid manifest: " + "; ".join(problems))
    frames = read_frames(root, manifest)
    imu = read_imu(root, manifest)
    if len(frames) != manifest.frames.n:
        raise ContractError(f"{root}: frames.n={manifest.frames.n} but index.csv has {len(frames)} rows")
    if len(imu) != manifest.imu.n:
        raise ContractError(f"{root}: imu.n={manifest.imu.n} but imu.csv has {len(imu)} rows")
    return CanonicalCapture(root=root, manifest=manifest, camera_model=model, frames=frames, imu=imu)


# ---------------------------------------------------------------------------
# validation against disk
# ---------------------------------------------------------------------------


def validate_capture(root: Path, *, check_images: bool = True) -> list[str]:
    """Full on-disk validation.  Returns a list of problems; an empty list means the capture is OK."""
    root = Path(root)
    problems: list[str] = []
    try:
        capture = read_capture(root)
    except Exception as e:                                  # noqa: BLE001 -- report, do not raise
        return [f"{type(e).__name__}: {e}"]
    m = capture.manifest
    problems += validate_manifest(m, capture.camera_model)

    for ref in m.outputs:
        p = root / ref.path
        if not p.exists():
            problems.append(f"output missing: {ref.path}")
            continue
        if p.stat().st_size != ref.bytes:
            problems.append(f"output size mismatch: {ref.path}")
        if sha256_file(p) != ref.sha256:
            problems.append(f"output sha256 mismatch: {ref.path}")

    ts = np.array([f.timestamp_ns for f in capture.frames], dtype=np.int64)
    if len(ts) > 1 and not np.all(np.diff(ts) > 0):
        problems.append("frame timestamps are not strictly increasing")
    its = capture.imu.timestamp_ns
    if len(its) > 1 and not np.all(np.diff(its) > 0):
        problems.append("imu timestamps are not strictly increasing")

    pinned = {}
    hp = root / m.frames.image_sha256_ref
    if hp.exists():
        for line in hp.read_text(encoding="utf-8").splitlines():
            if "  " in line:
                h, rel = line.split("  ", 1)
                pinned[rel] = h
    expect_w, expect_h = capture.camera_model.resolution_px
    for f in capture.frames:
        p = root / f.image_ref
        if not p.exists():
            problems.append(f"image missing: {f.image_ref}")
            continue
        if f.image_ref not in pinned:
            problems.append(f"image not listed in {m.frames.image_sha256_ref}: {f.image_ref}")
        elif check_images and sha256_file(p) != pinned[f.image_ref]:
            problems.append(f"image sha256 mismatch: {f.image_ref}")
        if check_images:
            try:
                w, h = png_gray8_size(p)
            except ContractError as e:
                problems.append(str(e))
                continue
            if [w, h] != [expect_w, expect_h]:
                problems.append(f"image {f.image_ref} is {w}x{h}, camera_model says {expect_w}x{expect_h}")
    extra = set(pinned) - {f.image_ref for f in capture.frames}
    if extra:
        problems.append(f"{m.frames.image_sha256_ref} lists {len(extra)} file(s) absent from the index")
    return problems


# ---------------------------------------------------------------------------
# format-level statistics (shared by the camera golden test and the boundary test)
# ---------------------------------------------------------------------------


def _stats(ts: np.ndarray) -> dict:
    if len(ts) < 2:
        return {"n": int(len(ts)), "t_first_ns": int(ts[0]) if len(ts) else None,
                "t_last_ns": int(ts[-1]) if len(ts) else None, "monotonic": True,
                "dt_ns_min": None, "dt_ns_max": None, "dt_ns_median": None, "rate_hz": None}
    d = np.diff(ts.astype(np.int64))
    span_ns = int(ts[-1] - ts[0])
    return {
        "n": int(len(ts)),
        "t_first_ns": int(ts[0]),
        "t_last_ns": int(ts[-1]),
        "monotonic": bool(np.all(d > 0)),
        "dt_ns_min": int(d.min()),
        "dt_ns_max": int(d.max()),
        "dt_ns_median": int(np.median(d)),
        "rate_hz": round((len(ts) - 1) * 1e9 / span_ns, 6) if span_ns else None,
    }


def capture_stats(capture: CanonicalCapture) -> dict:
    """Format-level statistics used as the golden signature of a capture.

    Deliberately contains nothing camera-specific: counts, timestamp statistics, image geometry and
    a pixel digest.  The pixel digest is taken over the DECODED arrays, so it survives a re-encode
    with a different zlib build.
    """
    fts = np.array([f.timestamp_ns for f in capture.frames], dtype=np.int64)
    out = {
        "schema_version": capture.manifest.schema_version,
        "frames": _stats(fts),
        "imu": _stats(capture.imu.timestamp_ns),
        "image_size_px": list(capture.camera_model.resolution_px),
        "frame_index_raw_first": int(capture.frames[0].frame_index_raw) if capture.frames else None,
        "frame_index_raw_last": int(capture.frames[-1].frame_index_raw) if capture.frames else None,
        "frame_step_raw": int(capture.manifest.frames.frame_step_raw),
    }
    if len(capture.imu):
        g = capture.imu.gyro_rad_s
        a = capture.imu.accel_m_s2
        out["imu_gyro_abs_max_rad_s"] = [round(float(x), 12) for x in np.abs(g).max(axis=0)]
        out["imu_accel_mean_m_s2"] = [round(float(x), 12) for x in a.mean(axis=0)]
        out["imu_accel_norm_mean_m_s2"] = round(float(np.linalg.norm(a, axis=1).mean()), 12)
    h = __import__("hashlib").sha256()
    for img in capture.images():
        h.update(np.ascontiguousarray(img).tobytes())
    out["pixel_sha256"] = h.hexdigest()
    return out


__all__ = [
    "CAMERA_MODEL_NAME", "FRAME_HASHES_NAME", "FRAME_INDEX_NAME", "IMU_DATA_NAME", "MANIFEST_NAME",
    "CanonicalCapture", "ImuStream",
    "encode_gray8_png", "decode_gray8_png", "read_gray8_png", "png_gray8_size",
    "write_capture", "read_capture", "read_manifest", "read_camera_model", "read_frames", "read_imu",
    "validate_capture", "capture_stats",
]
