import argparse
import os
import sys
import time

import numpy as np
from torch.utils.data import DataLoader

# Allow direct script execution from the debug/ folder.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.dataset_factory import DEFAULT_PATH, DatasetFactory


def passthrough_collate(batch):
    """Keep raw samples as-is because event arrays can vary in length."""
    return batch


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal N-ImageNet reader")
    parser.add_argument("--path", type=str, default=DEFAULT_PATH, help="Dataset root")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "npz", "lmdb"],
        default="auto",
        help="Storage backend to use",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "both"],
        default="train",
        help="Which split to read",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument(
        "--compare-backends",
        type=int,
        default=0,
        help="If >0, compare first N samples between npz and lmdb for each selected split",
    )
    return parser.parse_args()


def describe_events(events):
    total_events = int(events.shape[0])
    print(f"events shape: {events.shape} total_events={total_events}")
    print(f"events dtype: {events.dtype}")

    if total_events == 0:
        return

    if events.dtype.names and {"t", "x", "y", "p"}.issubset(set(events.dtype.names)):
        t = events["t"]
        x = events["x"]
        y = events["y"]
        p = events["p"]

        pos = int(np.sum(p > 0))
        neg = int(total_events - pos)
        print(
            f"t=[{int(t.min())}, {int(t.max())}] x=[{int(x.min())}, {int(x.max())}] "
            f"y=[{int(y.min())}, {int(y.max())}] p(+/-)=({pos}/{neg})"
        )


def run_split(path, train, batch_size, num_workers, num_steps, backend):
    split_name = "train" if train else "val"
    dataset = DatasetFactory.create(
        "nimagenet",
        train=train,
        transform=None,
        path=path,
        backend=backend,
    )

    print(f"\n=== split: {split_name} ===")
    print(f"backend: {backend}")
    print(f"dataset size: {len(dataset)}")
    if hasattr(dataset, "classes"):
        print(f"num classes: {len(dataset.classes)}")

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": train,
        "num_workers": num_workers,
        "collate_fn": passthrough_collate,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    loader = DataLoader(dataset, **loader_kwargs)

    start = time.time()
    for step, batch in enumerate(loader):
        events, target = batch[0]
        print(f"step {step}: batch_len={len(batch)} first_target={int(target)}")
        describe_events(events)
        print("-" * 20)

        if step + 1 >= num_steps:
            break

    elapsed = time.time() - start
    print(f"read {min(num_steps, len(loader))} steps in {elapsed:.2f}s")


def compare_backends(path, train, num_checks):
    split_name = "train" if train else "val"
    npz_ds = DatasetFactory.create("nimagenet", train=train, transform=None, path=path, backend="npz")
    lmdb_ds = DatasetFactory.create("nimagenet", train=train, transform=None, path=path, backend="lmdb")

    n = min(num_checks, len(npz_ds), len(lmdb_ds))
    print(f"\n=== parity check split: {split_name} samples={n} ===")
    if len(npz_ds) != len(lmdb_ds):
        raise RuntimeError(
            f"Length mismatch for split {split_name}: npz={len(npz_ds)} lmdb={len(lmdb_ds)}"
        )

    for idx in range(n):
        ev_npz, y_npz = npz_ds[idx]
        ev_lmdb, y_lmdb = lmdb_ds[idx]

        if int(y_npz) != int(y_lmdb):
            raise RuntimeError(f"Target mismatch at idx={idx}: npz={y_npz} lmdb={y_lmdb}")
        if ev_npz.dtype != ev_lmdb.dtype:
            raise RuntimeError(f"dtype mismatch at idx={idx}: npz={ev_npz.dtype} lmdb={ev_lmdb.dtype}")
        if ev_npz.shape != ev_lmdb.shape:
            raise RuntimeError(f"shape mismatch at idx={idx}: npz={ev_npz.shape} lmdb={ev_lmdb.shape}")
        if not np.array_equal(ev_npz, ev_lmdb):
            raise RuntimeError(f"event payload mismatch at idx={idx}")

    print("parity check passed")


def main():
    args = parse_args()

    if args.compare_backends > 0:
        if args.split in ("train", "both"):
            compare_backends(path=args.path, train=True, num_checks=args.compare_backends)
        if args.split in ("val", "both"):
            compare_backends(path=args.path, train=False, num_checks=args.compare_backends)

    if args.split in ("train", "both"):
        run_split(
            path=args.path,
            train=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_steps=args.num_steps,
            backend=args.backend,
        )

    if args.split in ("val", "both"):
        run_split(
            path=args.path,
            train=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_steps=args.num_steps,
            backend=args.backend,
        )


if __name__ == "__main__":
    main()

