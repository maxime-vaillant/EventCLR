#!/usr/bin/env python3
"""
Convert N-CARS .dat event files (Prophesee binary format) to .npz.

Binary format (Prophesee Version 2):
  Header: lines starting with '% ', followed by two metadata bytes (ev_type, ev_size).
  Each 8-byte event: [0:4] t uint32 LE, [4:8] addr uint32 LE
    addr bits 13:0  → x, bits 27:14 → y, bit 28 → p

Usage:
    python datasets/preprocess_ncars_dat.py /path/to/root --probe
    python datasets/preprocess_ncars_dat.py /path/to/root --dry-run
    python datasets/preprocess_ncars_dat.py /path/to/root
    python datasets/preprocess_ncars_dat.py /path/to/root --workers 4
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import numpy as np

X_MAX, Y_MAX = 119, 99


# ---------------------------------------------------------------------------
# Core reader  (Prophesee .dat format, Version 2)
# ---------------------------------------------------------------------------

def _parse_dat_header(f):
    """Skip '% …' comment lines, then read ev_type and ev_size metadata bytes."""
    num_comment_lines = 0
    while True:
        bod = f.tell()
        line = f.readline()
        if line.decode("latin-1")[:2] != "% ":
            break
        num_comment_lines += 1

    f.seek(bod, 0)
    if num_comment_lines > 0:
        ev_type = int(np.frombuffer(f.read(1), dtype=np.uint8)[0])
        ev_size = int(np.frombuffer(f.read(1), dtype=np.uint8)[0])
    else:
        ev_type, ev_size = 0, 8  # legacy files without comment lines
    return ev_type, ev_size


def load_events(path: str) -> dict:
    """
    Load a Prophesee/NCars .dat file.

    Returns dict with keys 't' (uint32), 'x' (uint16), 'y' (uint16), 'p' (uint8).
    """
    with open(path, "rb") as f:
        ev_type, ev_size = _parse_dat_header(f)
        if ev_size != 8:
            raise ValueError(f"{path}: unexpected ev_size={ev_size} (only 8-byte events supported)")
        raw = np.frombuffer(f.read(), dtype=np.dtype([("t", "<u4"), ("addr", "<u4")]))

    if len(raw) == 0:
        raise ValueError(f"{path}: file contains no events")

    addr = raw["addr"]
    return {
        "t": raw["t"],
        "x": (addr & 0x00003FFF).astype(np.uint16),          # bits 13:0
        "y": ((addr & 0x0FFFC000) >> 14).astype(np.uint16),  # bits 27:14
        "p": ((addr & 0x10000000) >> 28).astype(np.uint8),   # bit 28
    }


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

def probe(root: str) -> None:
    ncars_dir = os.path.join(root, "NCARS")
    sample_path = None
    for split in ("train", "test"):
        for cls in ("cars", "background"):
            cls_dir = os.path.join(ncars_dir, split, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.endswith(".dat"):
                    sample_path = os.path.join(cls_dir, fname)
                    break
            if sample_path:
                break
        if sample_path:
            break

    if not sample_path:
        sys.exit("[ERROR] No .dat files found.")

    print(f"Probing: {sample_path}\n")
    with open(sample_path, "rb") as f:
        content = f.read()

    print("=== ASCII header ===")
    pos = 0
    while pos < len(content):
        newline = content.find(b"\n", pos)
        end = newline + 1 if newline != -1 else len(content)
        if content[pos : pos + 1] == b"%":
            print(content[pos:end].decode("latin-1", errors="replace").rstrip())
            pos = end
        else:
            break
    print(f"\n(binary starts at byte {pos})\n")

    binary = content[pos:]
    print("=== First 128 binary bytes ===")
    for i in range(0, min(128, len(binary)), 4):
        chunk = binary[i : i + 4]
        hex_str = " ".join(f"{b:02x}" for b in chunk)
        u32 = int.from_bytes(chunk, "little")
        print(f"  offset {i:3d}: {hex_str}  →  uint32_le={u32:12d}  (0x{u32:08x})")

    events = load_events(sample_path)
    n = len(events["t"])
    print(f"\n{n} events loaded")
    print(f"  x: [{events['x'].min()}, {events['x'].max()}]")
    print(f"  y: [{events['y'].min()}, {events['y'].max()}]")
    print(f"  p: {np.unique(events['p']).tolist()}")
    print(f"  t: [{events['t'][0]}, {events['t'][-1]}]  ({(int(events['t'][-1]) - int(events['t'][0])) / 1e3:.1f} ms)")


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def _convert_file(dat_path: str, npz_path: str, overwrite: bool) -> str:
    if os.path.exists(npz_path) and not overwrite:
        return f"SKIP  {os.path.basename(dat_path)} (output exists)"
    try:
        events = load_events(dat_path)
    except Exception as exc:
        return f"ERROR {os.path.basename(dat_path)}: {exc}"

    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez_compressed(npz_path, **events)
    n = len(events["t"])
    dur_ms = (int(events["t"][-1]) - int(events["t"][0])) / 1e3
    return f"OK    {os.path.basename(dat_path)}  [{n} events, {dur_ms:.1f} ms]"


def preprocess(root: str, dry_run: bool, overwrite: bool = False, workers: int = 1) -> None:
    ncars_dir = os.path.join(root, "NCARS")
    if not os.path.isdir(ncars_dir):
        sys.exit(f"[ERROR] Not found: {ncars_dir}")

    jobs = []
    for split in ("train", "test"):
        for cls in ("cars", "background"):
            cls_dir = os.path.join(ncars_dir, split, cls)
            if not os.path.isdir(cls_dir):
                print(f"[WARN] Missing: {cls_dir}")
                continue
            dat_files = sorted(f for f in os.listdir(cls_dir) if f.endswith(".dat"))
            print(f"{split}/{cls}: {len(dat_files)} files")
            for fname in dat_files:
                dat_path = os.path.join(cls_dir, fname)
                npz_path = dat_path[:-4] + ".npz"
                jobs.append((dat_path, npz_path))

    if not jobs:
        sys.exit("[ERROR] No .dat files found.")

    if dry_run:
        out_of_range = 0
        for dat_path, _ in jobs:
            try:
                events = load_events(dat_path)
            except Exception as exc:
                print(f"  [error] {os.path.basename(dat_path)}: {exc}")
                continue
            x_max, y_max = int(events["x"].max()), int(events["y"].max())
            flag = ""
            if x_max > X_MAX or y_max > Y_MAX:
                out_of_range += 1
                flag = "  <-- OUT OF RANGE"
            print(f"  [dry-run] {os.path.basename(dat_path)}  {len(events['t'])} events  x=[0,{x_max}] y=[0,{y_max}]{flag}")
        if out_of_range:
            print(f"\n[WARN] {out_of_range}/{len(jobs)} files exceed [{X_MAX}x{Y_MAX}].")
        return

    if workers > 1 and len(jobs) > 1:
        worker_fn = partial(_convert_file, overwrite=overwrite)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(worker_fn, src, dst): src for src, dst in jobs}
            for fut in as_completed(futures):
                print(fut.result())
    else:
        for dat_path, npz_path in jobs:
            print(_convert_file(dat_path, npz_path, overwrite=overwrite))

    print(f"\nConverted {len(jobs)} files.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    if args.probe:
        probe(args.root)
    else:
        preprocess(args.root, dry_run=args.dry_run, overwrite=args.overwrite, workers=args.workers)