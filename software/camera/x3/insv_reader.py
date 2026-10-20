"""Insta360 INSV container reader: MP4 boxes, the private trailer, and its metadata record.

An ``.insv`` file is an MP4-compatible container with a private trailer appended at the end:

  * the last 32 bytes are the magic ``8db42d694ccc418790edff439fe026bf``;
  * the trailer header is the last 72 bytes: ``padding(32) + size(u32 LE) + version(u32 LE) + magic(32)``;
  * records are laid out from the trailer header BACKWARDS towards the start of the file.  Each
    record is ``data`` immediately followed by a 6-byte tail ``format(u8) + id(u8) + size(u32 LE)``;
  * record id 1 = metadata (protobuf-like), id 3 = gyro/IMU samples.

This module stays at the container level: it locates records and hands back their bytes, and it
decodes the metadata record's protobuf fields.  Turning the gyro record into physical units is
``imu_reader.py``'s job.  The trajectory reads frame times in ``software.trajectory.x3_extract``.

Everything here reads only the byte ranges the trailer advertises.  A recording is hundreds of MB
to several GB, so no function in this module reads a whole media file into memory.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

MAGIC = b"8db42d694ccc418790edff439fe026bf"

RECORD_NAMES = {0: "offsets", 1: "metadata", 3: "gyro", 4: "exposure",
                6: "timestamps", 7: "gps", 128: "timemap"}

RECORD_METADATA = 1
RECORD_GYRO = 3

#: Top-level metadata protobuf field numbers (no .proto available; field numbers are the contract).
FIELD_SERIAL = 1
FIELD_MODEL = 2
FIELD_FIRMWARE = 3
FIELD_OFFSET_V3 = 54            # lens parameters stored in the trailer; the trajectory does not read them
FIELD_FIRST_FRAME_TS_US = 24    # video t=0 expressed on the IMU clock, microseconds
FIELD_FIRST_FRAME_TS_MS = 36    # same value in milliseconds (cross-check only)
FIELD_GYRO_CFG_INFO = 65        # nested {1: acc_range_g, 2: gyro_range_dps}


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Chunked file hash.  Media files reach several GB; never read one whole into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# MP4 box structure
# ---------------------------------------------------------------------------


def walk_boxes(path: Path, limit: int = 40) -> list[dict]:
    """Walk the top-level MP4 boxes.  Non-box data before EOF is reported, not guessed at."""
    boxes: list[dict] = []
    size_total = Path(path).stat().st_size
    with open(path, "rb") as f:
        off = 0
        while off < size_total and len(boxes) < limit:
            f.seek(off)
            head = f.read(8)
            if len(head) < 8:
                break
            size = struct.unpack(">I", head[:4])[0]
            typ = head[4:8]
            if size == 1:
                size = struct.unpack(">Q", f.read(8))[0]
            if size < 8 or not (32 <= typ[0] <= 126):
                boxes.append({"offset": off, "note": "non-box data till EOF (trailer?)"})
                break
            boxes.append({"offset": off, "type": typ.decode("latin1"), "size": size})
            off += size
    return boxes


# ---------------------------------------------------------------------------
# private trailer
# ---------------------------------------------------------------------------


def parse_trailer(path: Path, max_records: int = 64, gyro_parser=None) -> dict:
    """Parse the private trailer.

    Returns ``{"present": False}`` when the magic is absent, otherwise
    ``{"present": True, "trailer_size", "version", "records": [...], "gyro": ...}`` where each
    record carries ``id``, ``name``, ``format``, ``size`` and ``offset`` (the absolute offset of the
    record payload in the file).

    ``gyro_parser`` decodes the gyro record payload; it defaults to
    :func:`software.camera.x3.imu_reader.parse_gyro` (imported lazily so the container reader has no
    module-level dependency on the IMU decoder).  Pass ``gyro_parser=False`` to skip gyro decoding
    entirely and only map the record layout.
    """
    path = Path(path)
    size_total = path.stat().st_size
    with open(path, "rb") as f:
        f.seek(size_total - 32)
        if f.read(32) != MAGIC:
            return {"present": False}
        f.seek(size_total - 72)
        header = f.read(72)
        trailer_size = struct.unpack("<I", header[32:36])[0]
        version = struct.unpack("<I", header[36:40])[0]
        out: dict = {"present": True, "trailer_size": trailer_size, "version": version,
                     "records": [], "gyro": None}

        if gyro_parser is None:
            from software.camera.x3.imu_reader import parse_gyro as gyro_parser  # noqa: PLC0415

        # records run from just before the trailer header towards the start of the file
        cursor = size_total - 72
        lo_bound = max(0, size_total - 72 - trailer_size - 4096)
        for _ in range(max_records):
            if cursor - 6 <= lo_bound:
                break
            f.seek(cursor - 6)
            fmt_b, id_b, rec_size = struct.unpack("<BBI", f.read(6))
            if rec_size == 0 or rec_size > cursor - 6:
                out["records"].append({"note": f"UNPARSED tail@{cursor}",
                                       "fmt": fmt_b, "id": id_b, "size": rec_size})
                break
            data_off = cursor - 6 - rec_size
            rec = {"id": id_b, "name": RECORD_NAMES.get(id_b, f"unknown_{id_b}"),
                   "format": fmt_b, "size": rec_size, "offset": data_off}
            if id_b == RECORD_GYRO and out["gyro"] is None and gyro_parser is not False:
                f.seek(data_off)
                out["gyro"] = gyro_parser(f.read(rec_size))
            out["records"].append(rec)
            cursor = data_off
        return out


def records_by_id(trailer: dict) -> dict[int, dict]:
    """``{record_id: record}`` for the records that carry a payload offset."""
    return {r["id"]: r for r in trailer.get("records", []) if "offset" in r}


def read_record_bytes(path: Path, record: dict) -> bytes:
    """Read exactly one record payload (seek + read of the advertised range)."""
    with open(path, "rb") as f:
        f.seek(record["offset"])
        return f.read(record["size"])


# ---------------------------------------------------------------------------
# metadata record (protobuf-like, no .proto)
# ---------------------------------------------------------------------------


def read_varint(b: bytes, i: int) -> tuple[int, int]:
    x = 0
    s = 0
    while True:
        c = b[i]
        i += 1
        x |= (c & 0x7F) << s
        s += 7
        if not (c & 0x80):
            return x, i


def parse_nested_varints(sub: bytes) -> dict:
    """Decode a nested message that contains only varint fields (gyro_cfg_info, dimension, ...)."""
    out = {}
    i = 0
    try:
        while i < len(sub):
            key, i = read_varint(sub, i)
            fn, wt = key >> 3, key & 7
            if wt != 0:
                return {}
            v, i = read_varint(sub, i)
            out[fn] = v
    except IndexError:
        return {}
    return out


def parse_metadata_record(data: bytes) -> dict:
    """Generic top-level protobuf decode (no .proto): ``{field_number: [values]}``.

    Length-delimited fields are decoded as UTF-8 when printable, otherwise kept as hex.
    """
    out: dict[int, list] = {}
    i = 0
    while i < len(data):
        key, i = read_varint(data, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = read_varint(data, i)
        elif wt == 1:
            v = struct.unpack("<d", data[i:i + 8])[0]
            i += 8
        elif wt == 2:
            ln, i = read_varint(data, i)
            sub = data[i:i + ln]
            i += ln
            try:
                s = sub.decode("utf-8")
                v = s if s.isprintable() else sub.hex()
            except UnicodeDecodeError:
                v = sub.hex()
            if fn == FIELD_GYRO_CFG_INFO:
                v = {"raw_hex": sub.hex(), "nested": parse_nested_varints(sub)}
        elif wt == 5:
            v = struct.unpack("<f", data[i:i + 4])[0]
            i += 4
        else:
            break
        out.setdefault(fn, []).append(v)
    return out


def read_metadata(path: Path, trailer: dict | None = None) -> dict:
    """Read and decode the metadata record of ``path``.  Returns ``{}`` when there is none."""
    trailer = trailer if trailer is not None else parse_trailer(path, gyro_parser=False)
    recs = records_by_id(trailer)
    if RECORD_METADATA not in recs:
        return {}
    return parse_metadata_record(read_record_bytes(path, recs[RECORD_METADATA]))


def metadata_summary(meta: dict) -> dict:
    """The identity / clock / lens fields this project uses, named instead of numbered."""
    cfg = (meta.get(FIELD_GYRO_CFG_INFO, [{}])[0] or {})
    nested = cfg.get("nested", {}) if isinstance(cfg, dict) else {}
    return {
        "serial": meta.get(FIELD_SERIAL, [None])[0],
        "model": meta.get(FIELD_MODEL, [None])[0],
        "firmware": meta.get(FIELD_FIRMWARE, [None])[0],
        "offset_v3": meta.get(FIELD_OFFSET_V3, [None])[0],
        "first_frame_timestamp_us": meta.get(FIELD_FIRST_FRAME_TS_US, [None])[0],
        "first_frame_timestamp_ms": meta.get(FIELD_FIRST_FRAME_TS_MS, [None])[0],
        "acc_range_g": nested.get(1),
        "gyro_range_dps": nested.get(2),
    }


__all__ = [
    "MAGIC", "RECORD_NAMES", "RECORD_GYRO", "RECORD_METADATA",
    "FIELD_SERIAL", "FIELD_MODEL", "FIELD_FIRMWARE", "FIELD_OFFSET_V3",
    "FIELD_FIRST_FRAME_TS_US", "FIELD_FIRST_FRAME_TS_MS", "FIELD_GYRO_CFG_INFO",
    "sha256_file", "walk_boxes", "parse_trailer", "records_by_id", "read_record_bytes",
    "read_varint", "parse_nested_varints", "parse_metadata_record", "read_metadata",
    "metadata_summary",
]
