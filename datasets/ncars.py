import os
import zipfile
import urllib.request
from typing import Optional, Callable

import numpy as np
import tonic.transforms as transforms
import torch
from torch.utils.data import Dataset


class NCARS(Dataset):
    """`N-CARS Dataset <https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/>`_

    A neuromorphic car/background classification dataset recorded with a Prophesee ATIS camera.
    Introduced in: "HATS: Histograms of Averaged Time Surfaces for Robust Event-based Object
    Classification", Sironi et al., CVPR 2018.

    The dataset contains short 100ms event clips at 120x100 resolution, split into two classes.
    Train/test splits are predefined (no manual splitting needed).

    .. note::
        If automatic download fails, manually download the preprocessed numpy archive from
        https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/
        and extract it so that ``<save_to>/NCARS/train/`` and ``<save_to>/NCARS/test/`` exist.

    Parameters:
        save_to (str): Root directory where the dataset will be stored.
        train (bool): If True, uses the training split; otherwise the test split.
        transform (callable, optional): Transform applied to the event array.
        target_transform (callable, optional): Transform applied to the class label.
        transforms (callable, optional): Transform applied jointly to events and label.
    """

    url = "https://www.prophesee.ai/wp-content/uploads/2019/11/N-CARS_preprocessed.zip"
    filename = "N-CARS_preprocessed.zip"
    folder_name = "NCARS"

    # (width, height, num_polarities) — matches tonic convention
    sensor_size = (120, 100, 2)

    # Canonical structured-array dtype expected by downstream tonic transforms
    dtype = np.dtype([("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", bool)])

    classes = {"background": 0, "cars": 1}
    num_classes = 2

    def __init__(
        self,
        save_to: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        self.save_to = os.path.expanduser(save_to)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms

        if not self._check_exists():
            self._download()

        self.data, self.targets = self._scan_files()

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        events = self._load_events(self.data[idx])
        target = self.targets[idx]

        if self.transforms is not None:
            events, target = self.transforms(events, target)
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return events, target

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _root(self) -> str:
        return os.path.join(self.save_to, self.folder_name)

    def _split_dir(self) -> str:
        return os.path.join(self._root, "train" if self.train else "test")

    def _check_exists(self) -> bool:
        return os.path.isdir(self._split_dir())

    def _download(self) -> None:
        os.makedirs(self.save_to, exist_ok=True)
        zip_path = os.path.join(self.save_to, self.filename)

        if not os.path.isfile(zip_path):
            print(f"Downloading N-CARS preprocessed dataset from:\n  {self.url}")
            try:
                urllib.request.urlretrieve(self.url, zip_path, self._reporthook)
                print()
            except Exception as exc:
                if os.path.isfile(zip_path):
                    os.remove(zip_path)
                raise RuntimeError(
                    f"Download failed: {exc}\n"
                    "Please download the dataset manually from:\n"
                    "  https://www.prophesee.ai/2020/01/24/"
                    "prophesee-gen1-automotive-detection-dataset/\n"
                    f"and extract it so that '{self._root}/train/' and "
                    f"'{self._root}/test/' exist."
                ) from exc

        print(f"Extracting {self.filename} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.save_to)

        if not self._check_exists():
            raise RuntimeError(
                f"Extraction finished but expected directory not found: {self._split_dir()}\n"
                "The archive structure may differ — please arrange the files manually so that "
                f"'{self._root}/train/background/', '{self._root}/train/cars/', "
                f"'{self._root}/test/background/', and '{self._root}/test/cars/' exist."
            )

    @staticmethod
    def _reporthook(count, block_size, total_size):
        if total_size > 0:
            pct = min(100, count * block_size * 100 // total_size)
            print(f"\r  {pct}%", end="", flush=True)

    def _scan_files(self):
        """Walk class subdirectories and collect (.npy path, label) pairs."""
        split_dir = self._split_dir()
        data, targets = [], []

        for class_name, class_idx in self.classes.items():
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                raise RuntimeError(
                    f"Class directory not found: {class_dir}\n"
                    f"Expected structure: {self._root}/{{train,test}}/{{background,cars}}/*.npy"
                )
            for fname in sorted(os.listdir(class_dir)):
                if fname.endswith(".npz") or fname.endswith(".npy"):
                    data.append(os.path.join(class_dir, fname))
                    targets.append(class_idx)

        if not data:
            raise RuntimeError(
                f"No .npz/.npy files found under {split_dir}. "
                "Make sure the dataset is correctly preprocessed."
            )

        return data, targets

    def _load_events(self, path: str) -> np.ndarray:
        """Load a .npz or .npy event file and normalise to the canonical dtype."""
        raw = np.load(path, allow_pickle=False)

        # .npz from preprocess_ncars_dat.py: separate t/x/y/p arrays
        if hasattr(raw, "files"):
            events = np.empty(len(raw["t"]), dtype=self.dtype)
            events["x"] = raw["x"].astype(np.int16)
            events["y"] = raw["y"].astype(np.int16)
            events["t"] = raw["t"].astype(np.int64)
            events["p"] = raw["p"].astype(bool)
            return events

        # Already in canonical form
        if raw.dtype == self.dtype:
            return raw

        # Remap field names ('ts' → 't' in some Prophesee releases)
        field_map = {name: name for name in raw.dtype.names or []}
        if "ts" in field_map and "t" not in field_map:
            field_map["ts"] = "t"

        if raw.dtype.names:
            events = np.empty(len(raw), dtype=self.dtype)
            for src, dst in field_map.items():
                if dst in self.dtype.names:
                    events[dst] = raw[src]
            return events

        # Fallback: assume columns are [x, y, t, p]
        events = np.empty(len(raw), dtype=self.dtype)
        events["x"] = raw[:, 0].astype(np.int16)
        events["y"] = raw[:, 1].astype(np.int16)
        events["t"] = raw[:, 2].astype(np.int64)
        events["p"] = raw[:, 3].astype(bool)
        return events