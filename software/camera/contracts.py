"""Camera/algorithm boundary contract (memo 08, section 2.3).

This module is the ONLY data contract between the camera layer
(``software/camera/<vendor>/``) and the algorithm layer (``software/algorithms/``).

Everything here is vendor neutral: no ``.insv``, no Insta360 file-naming rules, no hard-coded
4K60 / 1440 / 2880 constants.  A vendor backend fills these structures in; the algorithm layer
only ever reads them.

Design rules enforced by this module
------------------------------------
1. Units are explicit in every field NAME (``_ns``, ``_rad_s``, ``_m_s2``, ``_px``).  There is no
   implicit unit convention hidden in a comment.
2. Time is an ``int64`` nanosecond count.  Float seconds are never a storage format.  Timestamps
   MAY be negative (an IMU sample that precedes the frame-clock epoch keeps its real value instead
   of being clipped).
3. Every clock has a named domain and a named epoch (see :class:`ClockDomain`).  The offset between
   the frame clock and the IMU clock is an explicit field that is ``None`` when unknown -- it never
   silently defaults to 0.
4. Calibration/profile maturity travels WITH the data (``calibration_status``,
   ``camera_profile_status``).  A status may only be raised by an evidence file, never by editing a
   document or by a code default.
5. Anything that changed the pixels (half-frame selection, crop, resample) is recorded in
   :class:`PixelTransform` in ABSOLUTE source-frame coordinates, so a consumer can always map a
   stored pixel back to the raw sensor frame.

Disk format
-----------
See :data:`CAPTURE_LAYOUT` and ``software/camera/capture_io.py``.  The on-disk canonical capture is
readable with numpy + the standard library only (CSV/JSON/PNG); no camera SDK, no OpenCV, no PyAV.

Per-sample rows on disk carry only the fields that actually vary per sample
(``frames/index.csv`` -> ``timestamp_ns, frame_index_raw, image_ref``;
``imu/imu.csv`` -> ``timestamp_ns, gx, gy, gz, ax, ay, az``).  The capture-constant fields of
:class:`FrameSample` / :class:`ImuSample` live once in ``manifest.json``
(``frames.constant_fields`` / ``imu.constant_fields``) and are re-attached by the reader.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# versions
# ---------------------------------------------------------------------------

#: Version of the on-disk canonical capture layout AND of the dataclasses below.
#: Bump the MINOR part for backwards-compatible additive fields, the MAJOR part for anything that
#: an existing reader would mis-parse.
SCHEMA_VERSION = "umi.camera.capture/1.0.0"

#: Readers accept exactly these schema versions.
SUPPORTED_SCHEMA_VERSIONS = ("umi.camera.capture/1.0.0",)

CAPTURE_LAYOUT = """<capture>/
  manifest.json            # CaptureManifest
  camera_model.json        # CameraModel
  frames/
    index.csv              # timestamp_ns,frame_index_raw,image_ref   (header line starts with '#')
    images.sha256          # sha256sum-format digest of every image file, in index order
    <frame_index_raw:06d>.png   # 8-bit grayscale, non-interlaced, filter 0
  imu/
    imu.csv                # timestamp_ns,gyro_x_rad_s,...,accel_z_m_s2  (header starts with '#')
  reports/                 # OPTIONAL vendor self-check output; algorithms must not require it
"""

# ---------------------------------------------------------------------------
# closed vocabularies
# ---------------------------------------------------------------------------

#: What instant inside the exposure interval ``FrameSample.timestamp_ns`` refers to.
EXPOSURE_CONVENTIONS = ("mid_exposure", "start_of_readout", "unknown")

#: Projection model families.  The order of ``CameraModel.intrinsics`` is given by
#: ``CameraModel.intrinsic_names``, so a reader never has to guess a parameter order.
CAMERA_MODEL_TYPES = ("ucm_radtan", "kb4", "pinhole_radtan", "equirect")

#: Maturity of ``CameraModel``.  MAY ONLY be raised by an evidence file.
CALIBRATION_STATUSES = ("FACTORY_UNVERIFIED", "CANDIDATE", "VALIDATED")

#: Which half of a packed dual-fisheye frame the stored images come from.
LENS_HALVES = ("none", "left", "right")

#: Provenance of ``ClockSpec.imu_minus_frame_ns``.
CLOCK_OFFSET_SOURCES = ("measured", "asserted_by_device_metadata", "unknown")

#: Stored image encodings.  ``png_gray8`` = 8-bit grayscale PNG, colour type 0, non-interlaced.
IMAGE_FORMATS = ("png_gray8",)

#: Nominal standard gravity used to convert accelerometer LSB to m/s^2.  Recorded in the manifest so
#: a consumer never has to assume it.
STANDARD_GRAVITY_M_S2 = 9.80665


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Chunked SHA-256 of a file (sources can be multi-GB; never read one whole)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ContractError(ValueError):
    """Raised when a structure or a capture directory violates this contract."""


def _req(d: dict, key: str, ctx: str):
    if key not in d:
        raise ContractError(f"{ctx}: missing required field {key!r}")
    return d[key]


def _as_int_list(v, n: int, ctx: str) -> list[int]:
    if not isinstance(v, (list, tuple)) or len(v) != n:
        raise ContractError(f"{ctx}: expected a list of {n} integers, got {v!r}")
    return [int(x) for x in v]


def _as_float_list(v, ctx: str) -> list[float]:
    if not isinstance(v, (list, tuple)):
        raise ContractError(f"{ctx}: expected a list of floats, got {v!r}")
    return [float(x) for x in v]


def _in(value, allowed: tuple, ctx: str):
    if value not in allowed:
        raise ContractError(f"{ctx}: {value!r} not in {allowed}")
    return value


# ---------------------------------------------------------------------------
# per-sample records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameSample:
    """One image frame handed from the camera layer to the algorithm layer.

    Fields
    ------
    timestamp_ns
        int64 nanoseconds in the clock domain named by ``CaptureManifest.clock.frame``.
        Monotonically increasing within a capture.  The epoch is whatever that ClockDomain says;
        it is NOT assumed to be UTC, POSIX or zero-based.
    frame_index_raw
        Index of this frame in the RAW decoded media stream, before any frame skipping.  Two
        captures exported with different ``frame_step`` therefore still agree on this number.
    image_ref
        POSIX-style path of the image file relative to the capture root (e.g. ``frames/000042.png``).
        The reader never constructs this from a timestamp; it always uses this field.
    camera_id
        Stable id of the physical camera instance (body serial or a project alias).
    camera_model_id
        Points at ``CameraModel.model_id``; the projection model valid for THIS image's pixels.
    profile_id
        Recording profile id (recording mode + ROI decision), e.g. ``x3_360_5k7_30``.
    exposure_convention
        One of :data:`EXPOSURE_CONVENTIONS`.  ``unknown`` is a legitimate, honest value; it must not
        be silently promoted to ``mid_exposure``.
    schema_version
        Contract version this record was produced under.
    """

    timestamp_ns: int
    frame_index_raw: int
    image_ref: str
    camera_id: str
    camera_model_id: str
    profile_id: str
    exposure_convention: str = "unknown"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        _in(self.exposure_convention, EXPOSURE_CONVENTIONS, "FrameSample.exposure_convention")
        if int(self.frame_index_raw) < 0:
            raise ContractError(f"FrameSample.frame_index_raw must be >= 0, got {self.frame_index_raw}")
        if not self.image_ref or self.image_ref.startswith("/") or "\\" in self.image_ref:
            raise ContractError(f"FrameSample.image_ref must be a relative POSIX path, got {self.image_ref!r}")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ImuSample:
    """One inertial sample.

    Fields
    ------
    timestamp_ns
        int64 nanoseconds in the clock domain named by ``CaptureManifest.clock.imu``.  MAY be
        negative when the IMU stream starts before the frame-clock epoch -- that is data, not an
        error, and must not be clipped by the camera layer.
    gyro_rad_s
        Angular rate (x, y, z) in rad/s, in the frame named by ``frame_id``.
    accel_m_s2
        Specific force (x, y, z) in m/s^2, same frame.  The gravity constant used during the LSB
        conversion is recorded in ``CaptureManifest.imu.gravity_m_s2``.
    frame_id
        Name of the IMU body frame.  The axis convention belongs to the manifest
        (``imu.axes_note``); it is never left to a code comment.
    range_source
        Where the full-scale range / LSB scaling came from (e.g. a device metadata field id).
    """

    timestamp_ns: int
    gyro_rad_s: tuple[float, float, float]
    accel_m_s2: tuple[float, float, float]
    frame_id: str
    range_source: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        for name in ("gyro_rad_s", "accel_m_s2"):
            v = getattr(self, name)
            if len(v) != 3:
                raise ContractError(f"ImuSample.{name} must have 3 components, got {v!r}")
            if not all(math.isfinite(float(x)) for x in v):
                raise ContractError(f"ImuSample.{name} must be finite, got {v!r}")

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# camera model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PixelTransform:
    """Everything the camera layer did to the raw decoded frame to produce the stored image.

    All boxes are ABSOLUTE positions in the raw decoded frame, so a stored pixel can always be
    mapped back.  ``crop_xywh_abs is None`` means no crop.
    """

    source_size_px: list[int]             # [w, h] of the raw decoded frame
    lens_half: str                        # one of LENS_HALVES
    half_xywh: list[int]                  # the selected half (or the whole frame) in source coords
    crop_xywh_abs: list[int] | None       # crop box in source coords, or None
    output_size_px: list[int]             # [w, h] actually written to disk
    resampled: bool
    resample_interpolation: str = "none"  # e.g. "none", "cv2.INTER_AREA"

    def __post_init__(self):
        _in(self.lens_half, LENS_HALVES, "PixelTransform.lens_half")
        _as_int_list(self.source_size_px, 2, "PixelTransform.source_size_px")
        _as_int_list(self.half_xywh, 4, "PixelTransform.half_xywh")
        _as_int_list(self.output_size_px, 2, "PixelTransform.output_size_px")
        if self.crop_xywh_abs is not None:
            _as_int_list(self.crop_xywh_abs, 4, "PixelTransform.crop_xywh_abs")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PixelTransform":
        c = _req(d, "crop_xywh_abs", "PixelTransform")
        return cls(
            source_size_px=_as_int_list(_req(d, "source_size_px", "PixelTransform"), 2, "PixelTransform.source_size_px"),
            lens_half=_req(d, "lens_half", "PixelTransform"),
            half_xywh=_as_int_list(_req(d, "half_xywh", "PixelTransform"), 4, "PixelTransform.half_xywh"),
            crop_xywh_abs=None if c is None else _as_int_list(c, 4, "PixelTransform.crop_xywh_abs"),
            output_size_px=_as_int_list(_req(d, "output_size_px", "PixelTransform"), 2, "PixelTransform.output_size_px"),
            resampled=bool(_req(d, "resampled", "PixelTransform")),
            resample_interpolation=d.get("resample_interpolation", "none"),
        )


@dataclass(frozen=True)
class CameraModel:
    """Projection model valid for the STORED images (after half selection, crop and resample).

    ``intrinsics`` / ``distortion`` are plain float lists whose order is spelled out by
    ``intrinsic_names`` / ``distortion_names`` -- a reader never has to know a per-model convention.
    ``calibration_status`` records maturity and may only be raised by an evidence file.
    """

    model_id: str
    model_type: str
    resolution_px: list[int]              # [w, h] of the stored image
    intrinsic_names: list[str]
    intrinsics: list[float]
    distortion_names: list[str]
    distortion: list[float]
    calibration_status: str
    pixel_transform: PixelTransform
    roi_xywh: list[int] | None = None     # alias of pixel_transform.crop_xywh_abs, kept for memo 2.3
    lens_half: str = "none"
    projection_id: str = "identity"       # deterministic id of any reprojection/resampling recipe
    source: str = ""                      # where the numbers came from, verbatim
    note: str = ""
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        _in(self.model_type, CAMERA_MODEL_TYPES, "CameraModel.model_type")
        _in(self.calibration_status, CALIBRATION_STATUSES, "CameraModel.calibration_status")
        _in(self.lens_half, LENS_HALVES, "CameraModel.lens_half")
        _as_int_list(self.resolution_px, 2, "CameraModel.resolution_px")
        if len(self.intrinsic_names) != len(self.intrinsics):
            raise ContractError("CameraModel: intrinsic_names and intrinsics length mismatch")
        if len(self.distortion_names) != len(self.distortion):
            raise ContractError("CameraModel: distortion_names and distortion length mismatch")
        if list(self.resolution_px) != list(self.pixel_transform.output_size_px):
            raise ContractError(
                f"CameraModel.resolution_px {self.resolution_px} != pixel_transform.output_size_px "
                f"{self.pixel_transform.output_size_px}")
        if self.roi_xywh is not None:
            _as_int_list(self.roi_xywh, 4, "CameraModel.roi_xywh")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pixel_transform"] = self.pixel_transform.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CameraModel":
        roi = d.get("roi_xywh")
        return cls(
            model_id=_req(d, "model_id", "CameraModel"),
            model_type=_req(d, "model_type", "CameraModel"),
            resolution_px=_as_int_list(_req(d, "resolution_px", "CameraModel"), 2, "CameraModel.resolution_px"),
            intrinsic_names=list(_req(d, "intrinsic_names", "CameraModel")),
            intrinsics=_as_float_list(_req(d, "intrinsics", "CameraModel"), "CameraModel.intrinsics"),
            distortion_names=list(_req(d, "distortion_names", "CameraModel")),
            distortion=_as_float_list(_req(d, "distortion", "CameraModel"), "CameraModel.distortion"),
            calibration_status=_req(d, "calibration_status", "CameraModel"),
            pixel_transform=PixelTransform.from_dict(_req(d, "pixel_transform", "CameraModel")),
            roi_xywh=None if roi is None else _as_int_list(roi, 4, "CameraModel.roi_xywh"),
            lens_half=d.get("lens_half", "none"),
            projection_id=d.get("projection_id", "identity"),
            source=d.get("source", ""),
            note=d.get("note", ""),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )


# ---------------------------------------------------------------------------
# clock
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClockDomain:
    """A named time base.  ``epoch`` says what ``timestamp_ns == 0`` means, in words."""

    name: str
    epoch: str
    unit: str = "ns"
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ClockDomain":
        return cls(name=_req(d, "name", "ClockDomain"), epoch=_req(d, "epoch", "ClockDomain"),
                   unit=d.get("unit", "ns"), note=d.get("note", ""))


@dataclass(frozen=True)
class AppliedOffset:
    """An offset already baked into the stored timestamps.

    Recording these explicitly is what keeps "applied = 0" from becoming an unwritten assumption.
    """

    name: str
    value_ns: int
    applied_to: str        # "frames" | "imu" | "both"
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AppliedOffset":
        return cls(name=_req(d, "name", "AppliedOffset"), value_ns=int(_req(d, "value_ns", "AppliedOffset")),
                   applied_to=_req(d, "applied_to", "AppliedOffset"), reason=d.get("reason", ""))


@dataclass(frozen=True)
class ClockSpec:
    """The two clock domains of a capture and the known relation between them.

    ``imu_minus_frame_ns`` is ``imu_clock_reading - frame_clock_reading`` for the same physical
    instant.  ``None`` means UNKNOWN and must stay ``None``; a consumer that needs the value has to
    refuse or estimate it itself.  0 is only allowed together with an
    ``imu_minus_frame_source`` that justifies it.
    """

    frame: ClockDomain
    imu: ClockDomain
    imu_minus_frame_ns: int | None
    imu_minus_frame_source: str = "unknown"
    applied_offsets: list[AppliedOffset] = field(default_factory=list)

    def __post_init__(self):
        _in(self.imu_minus_frame_source, CLOCK_OFFSET_SOURCES, "ClockSpec.imu_minus_frame_source")
        if self.imu_minus_frame_ns is None and self.imu_minus_frame_source != "unknown":
            raise ContractError("ClockSpec: imu_minus_frame_ns is None but a source is claimed")
        if self.imu_minus_frame_ns is not None and self.imu_minus_frame_source == "unknown":
            raise ContractError("ClockSpec: imu_minus_frame_ns is set but its source is 'unknown'")

    def to_dict(self) -> dict:
        return {"frame": self.frame.to_dict(), "imu": self.imu.to_dict(),
                "imu_minus_frame_ns": self.imu_minus_frame_ns,
                "imu_minus_frame_source": self.imu_minus_frame_source,
                "applied_offsets": [o.to_dict() for o in self.applied_offsets]}

    @classmethod
    def from_dict(cls, d: dict) -> "ClockSpec":
        off = d.get("imu_minus_frame_ns")
        return cls(frame=ClockDomain.from_dict(_req(d, "frame", "ClockSpec")),
                   imu=ClockDomain.from_dict(_req(d, "imu", "ClockSpec")),
                   imu_minus_frame_ns=None if off is None else int(off),
                   imu_minus_frame_source=d.get("imu_minus_frame_source", "unknown"),
                   applied_offsets=[AppliedOffset.from_dict(x) for x in d.get("applied_offsets", [])])


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FileRef:
    """A file referenced by the manifest: a source input, or a capture output."""

    path: str          # source: as given by the operator; output: relative POSIX path in the capture
    bytes: int
    sha256: str
    role: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FileRef":
        return cls(path=_req(d, "path", "FileRef"), bytes=int(_req(d, "bytes", "FileRef")),
                   sha256=_req(d, "sha256", "FileRef"), role=d.get("role", ""))


@dataclass(frozen=True)
class FrameStreamSpec:
    """Where the frames live and which FrameSample fields are capture-constant."""

    index_ref: str                     # "frames/index.csv"
    image_dir: str                     # "frames"
    image_format: str                  # one of IMAGE_FORMATS
    image_sha256_ref: str              # "frames/images.sha256"
    n: int
    frame_step_raw: int = 1            # how many raw frames each stored frame advances
    constant_fields: dict = field(default_factory=dict)   # exposure_convention, ...

    def __post_init__(self):
        _in(self.image_format, IMAGE_FORMATS, "FrameStreamSpec.image_format")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FrameStreamSpec":
        return cls(index_ref=_req(d, "index_ref", "FrameStreamSpec"), image_dir=_req(d, "image_dir", "FrameStreamSpec"),
                   image_format=_req(d, "image_format", "FrameStreamSpec"),
                   image_sha256_ref=_req(d, "image_sha256_ref", "FrameStreamSpec"),
                   n=int(_req(d, "n", "FrameStreamSpec")), frame_step_raw=int(d.get("frame_step_raw", 1)),
                   constant_fields=dict(d.get("constant_fields", {})))


@dataclass(frozen=True)
class ImuStreamSpec:
    """Where the IMU lives, its units, and which ImuSample fields are capture-constant."""

    data_ref: str                      # "imu/imu.csv"
    n: int
    gravity_m_s2: float = STANDARD_GRAVITY_M_S2
    units: str = "gyro rad/s, accel m/s^2"
    axes_note: str = ""                # explicit axis convention; empty means UNDOCUMENTED
    constant_fields: dict = field(default_factory=dict)   # frame_id, range_source, ...
    scaling: dict = field(default_factory=dict)           # LSB/range provenance, verbatim
    integrity: dict = field(default_factory=dict)         # vendor integrity verdict, verbatim

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ImuStreamSpec":
        return cls(data_ref=_req(d, "data_ref", "ImuStreamSpec"), n=int(_req(d, "n", "ImuStreamSpec")),
                   gravity_m_s2=float(d.get("gravity_m_s2", STANDARD_GRAVITY_M_S2)),
                   units=d.get("units", "gyro rad/s, accel m/s^2"), axes_note=d.get("axes_note", ""),
                   constant_fields=dict(d.get("constant_fields", {})), scaling=dict(d.get("scaling", {})),
                   integrity=dict(d.get("integrity", {})))


@dataclass(frozen=True)
class Producer:
    """What produced the capture.  ``git_commit`` may be '' when the tree was dirty/unknown."""

    tool: str
    tool_version: str
    contracts_version: str = SCHEMA_VERSION
    git_commit: str = ""
    created_utc: str = ""
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Producer":
        return cls(tool=_req(d, "tool", "Producer"), tool_version=_req(d, "tool_version", "Producer"),
                   contracts_version=d.get("contracts_version", SCHEMA_VERSION), git_commit=d.get("git_commit", ""),
                   created_utc=d.get("created_utc", ""), note=d.get("note", ""))


@dataclass(frozen=True)
class CaptureManifest:
    """``manifest.json`` of a canonical capture -- the root of the contract.

    ``camera_profile_status`` carries the recording profile's maturity verbatim from the vendor
    profile table (e.g. ``CANDIDATE (...; no calibration yet)``).  Migrating a profile must NOT
    quietly turn a CANDIDATE default into a validated one.
    """

    schema_version: str
    capture_id: str
    camera_id: str
    camera_profile_id: str
    camera_profile_status: str
    camera_model_id: str
    camera_model_ref: str                 # "camera_model.json"
    clock: ClockSpec
    frames: FrameStreamSpec
    imu: ImuStreamSpec
    source_files: list[FileRef]
    outputs: list[FileRef]
    producer: Producer
    counts: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ContractError(f"CaptureManifest.schema_version {self.schema_version!r} not in "
                                f"{SUPPORTED_SCHEMA_VERSIONS}")

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "capture_id": self.capture_id,
            "camera_id": self.camera_id,
            "camera_profile_id": self.camera_profile_id,
            "camera_profile_status": self.camera_profile_status,
            "camera_model_id": self.camera_model_id,
            "camera_model_ref": self.camera_model_ref,
            "clock": self.clock.to_dict(),
            "frames": self.frames.to_dict(),
            "imu": self.imu.to_dict(),
            "source_files": [f.to_dict() for f in self.source_files],
            "outputs": [f.to_dict() for f in self.outputs],
            "producer": self.producer.to_dict(),
            "counts": dict(self.counts),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CaptureManifest":
        return cls(
            schema_version=_req(d, "schema_version", "CaptureManifest"),
            capture_id=_req(d, "capture_id", "CaptureManifest"),
            camera_id=_req(d, "camera_id", "CaptureManifest"),
            camera_profile_id=_req(d, "camera_profile_id", "CaptureManifest"),
            camera_profile_status=_req(d, "camera_profile_status", "CaptureManifest"),
            camera_model_id=_req(d, "camera_model_id", "CaptureManifest"),
            camera_model_ref=d.get("camera_model_ref", "camera_model.json"),
            clock=ClockSpec.from_dict(_req(d, "clock", "CaptureManifest")),
            frames=FrameStreamSpec.from_dict(_req(d, "frames", "CaptureManifest")),
            imu=ImuStreamSpec.from_dict(_req(d, "imu", "CaptureManifest")),
            source_files=[FileRef.from_dict(x) for x in d.get("source_files", [])],
            outputs=[FileRef.from_dict(x) for x in d.get("outputs", [])],
            producer=Producer.from_dict(_req(d, "producer", "CaptureManifest")),
            counts=dict(d.get("counts", {})),
            notes=list(d.get("notes", [])),
        )

    def to_json(self, indent: int = 1) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, sort_keys=False)

    @classmethod
    def from_json(cls, text: str) -> "CaptureManifest":
        return cls.from_dict(json.loads(text))


# ---------------------------------------------------------------------------
# structural validation (no disk access)
# ---------------------------------------------------------------------------


def validate_manifest(m: CaptureManifest, model: CameraModel | None = None) -> list[str]:
    """Return a list of contract problems.  Empty list == valid.  Never raises for content issues."""
    problems: list[str] = []
    if m.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        problems.append(f"manifest.schema_version {m.schema_version!r} unsupported")
    if not m.capture_id:
        problems.append("manifest.capture_id is empty")
    if m.frames.n < 0 or m.imu.n < 0:
        problems.append("negative stream length")
    if m.counts.get("n_frames") not in (None, m.frames.n):
        problems.append(f"counts.n_frames {m.counts.get('n_frames')} != frames.n {m.frames.n}")
    if m.counts.get("n_imu") not in (None, m.imu.n):
        problems.append(f"counts.n_imu {m.counts.get('n_imu')} != imu.n {m.imu.n}")
    if not m.imu.axes_note:
        problems.append("manifest.imu.axes_note is empty: the IMU axis convention is undocumented")
    if m.clock.imu_minus_frame_ns is None and m.clock.imu_minus_frame_source != "unknown":
        problems.append("manifest.clock: offset unknown but a source is claimed")
    if m.clock.imu_minus_frame_ns == 0 and m.clock.imu_minus_frame_source == "unknown":
        problems.append("manifest.clock: offset 0 with source 'unknown' -- an implicit default")
    for o in m.outputs:
        if o.path.startswith("/") or "\\" in o.path:
            problems.append(f"manifest.outputs: {o.path!r} is not a relative POSIX path")
        if len(o.sha256) != 64:
            problems.append(f"manifest.outputs: {o.path!r} has a malformed sha256")
    for s in m.source_files:
        if len(s.sha256) != 64:
            problems.append(f"manifest.source_files: {s.path!r} has a malformed sha256")
    if model is not None:
        if model.model_id != m.camera_model_id:
            problems.append(f"camera_model.model_id {model.model_id!r} != manifest.camera_model_id "
                            f"{m.camera_model_id!r}")
        if model.calibration_status not in CALIBRATION_STATUSES:
            problems.append(f"camera_model.calibration_status {model.calibration_status!r} unknown")
    return problems


__all__ = [
    "SCHEMA_VERSION", "SUPPORTED_SCHEMA_VERSIONS", "CAPTURE_LAYOUT",
    "EXPOSURE_CONVENTIONS", "CAMERA_MODEL_TYPES", "CALIBRATION_STATUSES", "LENS_HALVES",
    "CLOCK_OFFSET_SOURCES", "IMAGE_FORMATS", "STANDARD_GRAVITY_M_S2",
    "ContractError", "sha256_file", "sha256_bytes",
    "FrameSample", "ImuSample", "PixelTransform", "CameraModel",
    "ClockDomain", "AppliedOffset", "ClockSpec",
    "FileRef", "FrameStreamSpec", "ImuStreamSpec", "Producer", "CaptureManifest",
    "validate_manifest",
]
