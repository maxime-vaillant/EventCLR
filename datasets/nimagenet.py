import os
import json
import struct
from typing import Callable, Optional

import numpy as np

from tonic.dataset import Dataset

import lmdb

class NImageNet(Dataset):
    """N-ImageNet dataset wrapper following Tonic-style conventions.

    Expected folder layout:

        NImageNet/
            training/
                <class_name>/
                    sample_1.npz
                    sample_2.npz
                    ...
            validation/
                extracted_val/
                    <class_name>/
                        sample_1.npz
                        sample_2.npz
                        ...

    Parameters:
        save_to: Base folder containing `NImageNet`.
        train: If True load from `training`, otherwise from
            `validation/extracted_val`.
        transform: Transform applied to events only.
        target_transform: Transform applied to label only.
        transforms: Transform applied jointly to (events, label).
        max_events: If set, skip samples with more than this many events.
    """

    dtype = np.dtype(
        [("t", np.uint64), ("x", np.uint16), ("y", np.uint16), ("p", bool)]
    )
    ordering = dtype.names
    # Keep this explicit for the factory; can be overridden per instance if needed.
    sensor_size = (640, 480, 2)
    lmdb_schema_version = 1

    _LMDB_SAMPLE_PREFIX = b"sample/"
    _LMDB_INDEX_PREFIX = b"index/"
    _LMDB_HEADER = struct.Struct("<II")  # target(uint32), n_events(uint32)

    def __init__(
        self,
        save_to: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        max_events: Optional[int] = None,
        backend: str = "auto",
        lmdb_path: Optional[str] = None,
    ):
        super().__init__(
            save_to,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )

        self.train = train
        self.max_events = max_events
        self.backend = backend.lower()
        if self.backend not in {"auto", "npz", "lmdb"}:
            raise ValueError("backend must be one of: auto, npz, lmdb")

        split_name = "train" if train else "val"
        split_dir = "training" if train else os.path.join("validation", "extracted_val")
        self.split_path = os.path.join(self.location_on_system, self.folder_name, split_dir)
        self.lmdb_path = lmdb_path or os.path.join(
            self.location_on_system,
            self.folder_name,
            "lmdb",
            f"{split_name}.lmdb",
        )
        self._lmdb_env = None

        use_lmdb = self.backend == "lmdb" or (
            self.backend == "auto" and self._lmdb_exists(self.lmdb_path)
        )
        if use_lmdb:
            self._init_from_lmdb()
        else:
            self._init_from_npz()

    def _init_from_npz(self):
        if not os.path.isdir(self.split_path):
            raise FileNotFoundError(
                f"N-ImageNet split folder not found: {self.split_path}. "
                "Expected `NImageNet/training/...` or "
                "`NImageNet/validation/extracted_val/...`."
            )

        self.classes = sorted(
            [
                class_name
                for class_name in os.listdir(self.split_path)
                if os.path.isdir(os.path.join(self.split_path, class_name))
            ]
        )
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}

        self.data = []
        self.targets = []

        for class_name in self.classes:
            class_dir = os.path.join(self.split_path, class_name)
            for filename in sorted(os.listdir(class_dir)):
                if not filename.endswith(".npz"):
                    continue

                file_path = os.path.join(class_dir, filename)
                if self.max_events is not None:
                    event_count = self._count_events(file_path)
                    if event_count > self.max_events:
                        continue

                self.data.append(file_path)
                self.targets.append(self.class_to_idx[class_name])

        if len(self.data) == 0:
            if self.max_events is None:
                raise RuntimeError(f"No .npz files found under {self.split_path}.")
            raise RuntimeError(
                f"No .npz files under {self.split_path} satisfy max_events={self.max_events}."
            )

    def _init_from_lmdb(self):
        if lmdb is None:
            raise ImportError(
                "LMDB backend requested but package `lmdb` is not installed. "
                "Install it with `pip install lmdb`."
            )
        if not self._lmdb_exists(self.lmdb_path):
            raise FileNotFoundError(
                f"LMDB split not found at {self.lmdb_path}. "
                "Expected an LMDB directory containing data.mdb."
            )

        env = self._open_lmdb_env()
        with env.begin(write=False) as txn:
            version_raw = txn.get(b"meta/version")
            if version_raw is not None and int(version_raw.decode("utf-8")) != self.lmdb_schema_version:
                raise RuntimeError(
                    f"Unsupported NImageNet LMDB schema version: {version_raw!r}"
                )

            classes_raw = txn.get(b"meta/classes")
            class_to_idx_raw = txn.get(b"meta/class_to_idx")
            length_raw = txn.get(b"meta/length")
            if classes_raw is None or class_to_idx_raw is None or length_raw is None:
                raise RuntimeError(
                    "Invalid NImageNet LMDB metadata. Missing one of: "
                    "meta/classes, meta/class_to_idx, meta/length."
                )

            self.classes = json.loads(classes_raw.decode("utf-8"))
            parsed_class_to_idx = json.loads(class_to_idx_raw.decode("utf-8"))
            self.class_to_idx = {k: int(v) for k, v in parsed_class_to_idx.items()}
            total_len = int(length_raw.decode("utf-8"))

            self.data = []
            self.targets = []
            for idx in range(total_len):
                index_blob = txn.get(self._lmdb_index_key(idx))
                if index_blob is None:
                    raise RuntimeError(f"Missing LMDB index key for sample {idx}.")
                target, event_count = self._LMDB_HEADER.unpack(index_blob)
                if self.max_events is not None and event_count > self.max_events:
                    continue
                self.data.append(idx)
                self.targets.append(int(target))

        if len(self.data) == 0:
            if self.max_events is None:
                raise RuntimeError(f"No samples available in {self.lmdb_path}.")
            raise RuntimeError(
                f"No samples in {self.lmdb_path} satisfy max_events={self.max_events}."
            )

    @staticmethod
    def _lmdb_exists(path: str) -> bool:
        return os.path.isfile(os.path.join(path, "data.mdb"))

    @classmethod
    def _lmdb_sample_key(cls, idx: int) -> bytes:
        return cls._LMDB_SAMPLE_PREFIX + f"{idx:09d}".encode("ascii")

    @classmethod
    def _lmdb_index_key(cls, idx: int) -> bytes:
        return cls._LMDB_INDEX_PREFIX + f"{idx:09d}".encode("ascii")

    def _open_lmdb_env(self):
        if self._lmdb_env is None:
            assert lmdb is not None, "lmdb package is required for LMDB backend"
            self._lmdb_env = lmdb.open(
                self.lmdb_path,
                subdir=True,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=512,
            )
        return self._lmdb_env

    def __getstate__(self):
        # Avoid sharing LMDB handles across DataLoader worker forks.
        state = self.__dict__.copy()
        state["_lmdb_env"] = None
        return state

    def __getitem__(self, index):
        """Return a tuple of (events, target)."""
        if self.data and isinstance(self.data[index], int):
            events = self._load_events_lmdb(self.data[index])
        else:
            events = self._load_events(self.data[index])
        target = self.targets[index]

        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transforms is not None:
            events, target = self.transforms(events, target)

        return events, target

    def __len__(self):
        return len(self.data)

    @classmethod
    def _build_events_array(cls, t, x, y, p):
        if not (len(t) == len(x) == len(y) == len(p)):
            raise ValueError("Event arrays t/x/y/p must all have the same length.")

        events = np.empty(len(t), dtype=cls.dtype)
        events["t"] = np.asarray(t, dtype=np.uint64)
        events["x"] = np.asarray(x, dtype=np.uint16)
        events["y"] = np.asarray(y, dtype=np.uint16)
        events["p"] = np.asarray(p) > 0
        return events

    @classmethod
    def _from_packed_events(cls, packed):
        # Structured events with named columns.
        if getattr(packed, "dtype", None) is not None and packed.dtype.names is not None:
            names = set(packed.dtype.names)
            if {"t", "x", "y", "p"}.issubset(names):
                return cls._build_events_array(
                    packed["t"], packed["x"], packed["y"], packed["p"]
                )

        # Dense events with four columns [t, x, y, p].
        if getattr(packed, "ndim", None) == 2 and packed.shape[1] >= 4:
            return cls._build_events_array(
                packed[:, 0], packed[:, 1], packed[:, 2], packed[:, 3]
            )

        # Object arrays may wrap dicts/records after np.savez.
        if getattr(packed, "dtype", None) is not None and packed.dtype == object:
            if packed.shape == ():
                obj = packed.item()
            elif packed.size == 1:
                obj = packed.reshape(()).item()
            else:
                obj = None

            if isinstance(obj, dict) and {"t", "x", "y", "p"}.issubset(obj.keys()):
                return cls._build_events_array(obj["t"], obj["x"], obj["y"], obj["p"])

        return None

    @classmethod
    def _count_events(cls, file_path: str) -> int:
        # Parse once using the same decoder used by __getitem__ for reliable counts.
        return len(cls._load_events(file_path))

    @classmethod
    def encode_lmdb_sample(cls, events: np.ndarray, target: int) -> bytes:
        t = np.asarray(events["t"], dtype=np.uint64)
        x = np.asarray(events["x"], dtype=np.uint16)
        y = np.asarray(events["y"], dtype=np.uint16)
        p = np.asarray(events["p"], dtype=np.uint8)
        if not (len(t) == len(x) == len(y) == len(p)):
            raise ValueError("Event arrays t/x/y/p must all have the same length.")

        header = cls._LMDB_HEADER.pack(int(target), int(len(t)))
        return b"".join([header, t.tobytes(), x.tobytes(), y.tobytes(), p.tobytes()])

    @classmethod
    def decode_lmdb_sample(cls, blob: bytes):
        target, n_events = cls._LMDB_HEADER.unpack_from(blob, 0)
        offset = cls._LMDB_HEADER.size

        t_nbytes = n_events * np.dtype(np.uint64).itemsize
        x_nbytes = n_events * np.dtype(np.uint16).itemsize
        y_nbytes = n_events * np.dtype(np.uint16).itemsize
        p_nbytes = n_events * np.dtype(np.uint8).itemsize

        t = np.frombuffer(blob, dtype=np.uint64, count=n_events, offset=offset)
        offset += t_nbytes
        x = np.frombuffer(blob, dtype=np.uint16, count=n_events, offset=offset)
        offset += x_nbytes
        y = np.frombuffer(blob, dtype=np.uint16, count=n_events, offset=offset)
        offset += y_nbytes
        p = np.frombuffer(blob, dtype=np.uint8, count=n_events, offset=offset)
        offset += p_nbytes

        if offset != len(blob):
            raise ValueError("Corrupt LMDB sample: trailing bytes detected.")

        events = np.empty(n_events, dtype=cls.dtype)
        events["t"] = t
        events["x"] = x
        events["y"] = y
        events["p"] = p > 0
        return events, int(target)

    def _load_events_lmdb(self, sample_idx: int):
        env = self._open_lmdb_env()
        with env.begin(write=False) as txn:
            blob = txn.get(self._lmdb_sample_key(sample_idx))
        if blob is None:
            raise KeyError(f"Missing LMDB sample key: {sample_idx}")
        events, _ = self.decode_lmdb_sample(blob)
        return events

    @classmethod
    def _load_events(cls, file_path: str):
        with np.load(file_path, allow_pickle=False) as sample:
            keys = set(sample.files)

            # Preferred key in this dataset export.
            if "event_data" in keys:
                parsed = cls._from_packed_events(sample["event_data"])
                if parsed is not None:
                    return parsed

            # Most common layout in neuromorphic .npz files.
            if {"t", "x", "y", "p"}.issubset(keys):
                return cls._build_events_array(
                    sample["t"], sample["x"], sample["y"], sample["p"]
                )

            # Fallback for single packed array under `events`.
            if "events" in keys:
                parsed = cls._from_packed_events(sample["events"])
                if parsed is not None:
                    return parsed

        raise ValueError(
            f"Unsupported event format in {file_path}. "
            "Expected `event_data`, keys (`t`, `x`, `y`, `p`) or `events` with 4 columns."
        )
