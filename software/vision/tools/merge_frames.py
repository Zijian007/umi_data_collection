"""merge_frames.py --out merged.csv --src label:priority:frames.csv [--src ...]
Per frame, keep using the current source while it has a valid pose (ok=1, state=2); otherwise switch to the valid source with the
highest priority (e.g. keyframe count of its map). Writes frames.csv columns + 'map' (= source label); invalid frames ok=0.
Chains must be split wherever 'map' changes (different coordinate frames). The fusion does that."""
import argparse
import csv


def merge(sources, output):
    """合并已排序优先级的定位观测；同来源有效时保持来源。"""
    srcs, out = sources, output
    data = []
    for lab, pri, path in srcs:
        d = {}
        for r in csv.DictReader(open(path)):
            d[int(r["frame_idx"])] = r
        data.append((lab, float(pri), d))
    data.sort(key=lambda x: -x[1])
    nfr = max(max(d) for _, _, d in data) + 1
    cols = ["frame_idx", "t", "state", "ok", "tx", "ty", "tz", "qx", "qy", "qz", "qw", "map"]
    cur = None; counts = {lab: 0 for lab, _, _ in data}; switches = 0
    with open(out, "w", newline="") as fo:
        w = csv.writer(fo); w.writerow(cols)
        for k in range(nfr):
            valid = {lab: d[k] for lab, _, d in data if k in d and d[k]["ok"] == "1" and d[k]["state"] == "2"}
            if cur not in valid:
                nxt = next((lab for lab, _, _ in data if lab in valid), None)
                if nxt is not None and cur is not None and nxt != cur:
                    switches += 1
                cur = nxt if nxt is not None else cur
            if cur in valid:
                r = valid[cur]; counts[cur] += 1
                w.writerow([k, r["t"], 2, 1] + [r[c] for c in ("tx", "ty", "tz", "qx", "qy", "qz", "qw")] + [cur])
            else:
                anyr = next((d[k] for _, _, d in data if k in d), None)
                w.writerow([k, anyr["t"] if anyr else "%.9f" % (k / 29.97), 1, 0, 0, 0, 0, 0, 0, 0, 1, ""])
    return dict(frames=nfr, counts=counts, switches=switches)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--src', action='append', required=True)
    args = parser.parse_args(argv)
    result = merge([item.split(':', 2) for item in args.src], args.out)
    print('merged', args.out, 'frames', result['frames'], 'valid per source', result['counts'], 'switches', result['switches'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
