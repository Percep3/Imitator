"""Read-only, dependency-free TensorBoard scalar dumper (stdlib only).

Usage: python tb_scalars.py <events-file-or-dir> [tag_substring ...]
Prints, per tag: n, first, min(step), max(step), last. No CRC check.
"""
import struct, sys
from pathlib import Path


def varint(b, i):
    r = s = 0
    while True:
        c = b[i]; i += 1
        r |= (c & 0x7F) << s; s += 7
        if c < 0x80:
            return r, i


def fields(b):
    i = 0
    while i < len(b):
        key, i = varint(b, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = varint(b, i)
        elif wt == 1:
            v = b[i:i + 8]; i += 8
        elif wt == 2:
            n, i = varint(b, i); v = b[i:i + n]; i += n
        elif wt == 5:
            v = b[i:i + 4]; i += 4
        else:
            return
        yield fn, wt, v


def events(path):
    data = Path(path).read_bytes()
    i = 0
    while i + 12 <= len(data):
        (n,) = struct.unpack("<Q", data[i:i + 8]); i += 12
        yield data[i:i + n]; i += n + 4


def scalars(path):
    for ev in events(path):
        step = 0; summ = None
        for fn, wt, v in fields(ev):
            if fn == 2 and wt == 0:
                step = v
            elif fn == 5 and wt == 2:
                summ = v
        if summ is None:
            continue
        for fn, wt, val in fields(summ):
            if fn != 1:
                continue
            tag = None; x = None
            for f2, w2, v2 in fields(val):
                if f2 == 1:
                    tag = v2.decode("utf-8", "replace")
                elif f2 == 2 and w2 == 5:
                    x = struct.unpack("<f", v2)[0]
            if tag is not None and x is not None:
                yield tag, step, x


if __name__ == "__main__":
    root = Path(sys.argv[1]); subs = sys.argv[2:]
    files = [root] if root.is_file() else sorted(root.rglob("events.out.tfevents.*"))
    for f in files:
        rows = {}
        for tag, step, x in scalars(f):
            if not subs or any(s.lower() in tag.lower() for s in subs):
                rows.setdefault(tag, []).append((step, x))
        if not rows:
            continue
        print(f"## {f}")
        for tag, r in sorted(rows.items()):
            mn = min(r, key=lambda t: t[1]); mx = max(r, key=lambda t: t[1])
            print(f"{tag}: n={len(r)} first={r[0][1]:.5g}@{r[0][0]} min={mn[1]:.5g}@{mn[0]} "
                  f"max={mx[1]:.5g}@{mx[0]} last={r[-1][1]:.5g}@{r[-1][0]}")
