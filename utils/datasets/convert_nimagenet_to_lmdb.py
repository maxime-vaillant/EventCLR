import argparse
import json
import os
import shutil
import sys
import time

# Allow direct script execution from the debug/ folder.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import lmdb

from datasets.nimagenet import NImageNet


def parse_args():
    parser = argparse.ArgumentParser(description="Convert N-ImageNet .npz split(s) to LMDB")
    parser.add_argument("--path", type=str, default=os.path.expanduser("~/data"), help="Dataset root")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "both"],
        default="both",
        help="Split to convert",
    )
    parser.add_argument(
        "--map-size-gb",
        type=float,
        default=64.0,
        help="LMDB map size in GiB",
    )
    parser.add_argument(
        "--commit-interval",
        type=int,
        default=512,
        help="Number of samples per write transaction commit",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing split LMDB before writing",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optionally keep only samples with <= max events",
    )
    return parser.parse_args()


def _split_lmdb_path(path: str, train: bool) -> str:
    split_name = "train" if train else "val"
    return os.path.join(path, "NImageNet", "lmdb", f"{split_name}.lmdb")


def _prepare_output(path: str, overwrite: bool):
    if os.path.isdir(path):
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {path}. Use --overwrite to rebuild."
            )
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def convert_split(path: str, train: bool, map_size_gb: float, commit_interval: int, overwrite: bool, max_events: int | None):
    if lmdb is None:
        raise ImportError("Package `lmdb` is required. Install with `pip install lmdb`.")
    assert lmdb is not None

    split_name = "train" if train else "val"
    output_path = _split_lmdb_path(path, train)
    _prepare_output(output_path, overwrite=overwrite)

    dataset = NImageNet(
        save_to=path,
        train=train,
        transform=None,
        backend="npz",
        max_events=max_events,
    )

    map_size = int(map_size_gb * (1024 ** 3))
    env = lmdb.open(
        output_path,
        subdir=True,
        map_size=map_size,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
        map_async=True,
        writemap=False,
    )

    print(f"\n=== converting split: {split_name} ===")
    print(f"samples: {len(dataset)} classes: {len(dataset.classes)}")
    print(f"output: {output_path}")

    started = time.time()
    txn = env.begin(write=True)
    total_events = 0

    for idx, (file_path, target) in enumerate(zip(dataset.data, dataset.targets)):
        events = NImageNet._load_events(file_path)
        total_events += int(events.shape[0])

        sample_blob = NImageNet.encode_lmdb_sample(events, target)
        index_blob = NImageNet._LMDB_HEADER.pack(int(target), int(events.shape[0]))

        txn.put(NImageNet._lmdb_sample_key(idx), sample_blob)
        txn.put(NImageNet._lmdb_index_key(idx), index_blob)

        if (idx + 1) % commit_interval == 0:
            txn.commit()
            txn = env.begin(write=True)
            elapsed = time.time() - started
            print(f"written {idx + 1}/{len(dataset)} samples in {elapsed:.1f}s")

    txn.put(b"meta/version", str(NImageNet.lmdb_schema_version).encode("utf-8"))
    txn.put(b"meta/split", split_name.encode("utf-8"))
    txn.put(b"meta/length", str(len(dataset)).encode("utf-8"))
    txn.put(b"meta/classes", json.dumps(dataset.classes).encode("utf-8"))
    txn.put(
        b"meta/class_to_idx",
        json.dumps(dataset.class_to_idx, sort_keys=True).encode("utf-8"),
    )
    txn.put(b"meta/total_events", str(total_events).encode("utf-8"))
    txn.commit()

    env.sync()
    env.close()

    elapsed = time.time() - started
    events_per_sec = total_events / max(elapsed, 1e-6)
    print(
        f"done split={split_name} samples={len(dataset)} total_events={total_events} "
        f"elapsed={elapsed:.1f}s events/s={events_per_sec:,.0f}"
    )


def main():
    args = parse_args()

    if args.split in ("train", "both"):
        convert_split(
            path=args.path,
            train=True,
            map_size_gb=args.map_size_gb,
            commit_interval=args.commit_interval,
            overwrite=args.overwrite,
            max_events=args.max_events,
        )

    if args.split in ("val", "both"):
        convert_split(
            path=args.path,
            train=False,
            map_size_gb=args.map_size_gb,
            commit_interval=args.commit_interval,
            overwrite=args.overwrite,
            max_events=args.max_events,
        )


if __name__ == "__main__":
    main()


