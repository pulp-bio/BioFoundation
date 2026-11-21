# SPDX-License-Identifier: Apache-2.0
"""Pack an ARES bin/ directory into the read-only FS blob read by pmsis_r5.c.
Layout: b"ARESFS01", u32 count, count x {char name[120], u32 offset, u32 size}, then file data
(offsets from the blob start, 64-byte aligned).  Usage: python3 pack_fs.py <bin_dir> <out.bin>"""
import os, struct, sys
src, out = sys.argv[1], sys.argv[2]
names = sorted(f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f)))
hdr = 12 + 128 * len(names)
off = (hdr + 63) & ~63
ents, datas = [], []
for n in names:
    assert len(n.encode()) < 120, n
    d = open(os.path.join(src, n), "rb").read()
    ents.append(struct.pack("<120sII", n.encode(), off, len(d)))
    datas.append((off, d))
    off = (off + len(d) + 63) & ~63
blob = bytearray(off)
blob[:12] = b"ARESFS01" + struct.pack("<I", len(names))
blob[12:12 + 128 * len(names)] = b"".join(ents)
for o, d in datas:
    blob[o:o + len(d)] = d
open(out, "wb").write(blob)
print(f"{out}: {len(names)} files, {len(blob) / 2**20:.2f} MiB")
