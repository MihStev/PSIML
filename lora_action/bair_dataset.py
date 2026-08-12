#!/usr/bin/env python
"""
Dataset for the real BAIR LoRA training loop. Extends the repo's existing
CameraLatentLMDBDataset (wan_utils/dataset.py) with the `actions` field
written by build_bair_lmdb.py, and pre-computes the per-latent-frame action
alignment decided in CLAUDE.md ("ARHITEKTONSKA ODLUKA", 12.08):

    latent 0        -> no preceding action (zeros)
    latent i (i>=1) -> raw actions[4*(i-1) : 4*i], FLATTENED (16 dims, not
                        averaged -- mean-pooling was shown to be disqualifying,
                        see CLAUDE.md)

Camera fields (viewmats/Ks) are still read by the parent class but are dummy
identity placeholders (BAIR has no camera) -- kept only for LMDB schema
compatibility, not used by the action-conditioning path.
"""
import sys

sys.path.insert(0, "/home/mls10/minWM-dawidzard/Wan21")

import numpy as np
import torch

from wan_utils.dataset import CameraLatentLMDBDataset
from wan_utils.lmdb_ import get_array_shape_from_lmdb, retrieve_row_from_lmdb

ACTIONS_PER_LATENT_DIM = 16  # 4 raw BAIR actions x 4 dims, flattened


class BairActionLatentDataset(CameraLatentLMDBDataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        super().__init__(data_path, max_pair)
        if self._sharded:
            self._actions_shapes = [
                get_array_shape_from_lmdb(env, "actions") for env in self.envs
            ]
        else:
            self.actions_shape = get_array_shape_from_lmdb(self.env, "actions")

    def _raw_actions(self, idx):
        if self._sharded:
            sid, local_idx = self.index[idx]
            env = self.envs[sid]
            shape = self._actions_shapes[sid]
            return retrieve_row_from_lmdb(env, "actions", np.float32, local_idx, shape=shape[1:])
        return retrieve_row_from_lmdb(self.env, "actions", np.float32, idx, shape=self.actions_shape[1:])

    @staticmethod
    def _align_to_latent_frames(raw_actions_30x4: np.ndarray, n_latent_frames: int) -> np.ndarray:
        out = np.zeros((n_latent_frames, ACTIONS_PER_LATENT_DIM), dtype=np.float32)
        for i in range(1, n_latent_frames):
            start = 4 * (i - 1)
            chunk = raw_actions_30x4[start:start + 4].flatten()
            out[i, :len(chunk)] = chunk  # defensive: zero-pad if fewer than 4 remain
        return out

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        raw_actions = self._raw_actions(idx)  # (30, 4)
        n_latent_frames = item["clean_latent"].shape[0]
        actions_per_latent = self._align_to_latent_frames(raw_actions, n_latent_frames)
        item["actions_per_latent"] = torch.tensor(actions_per_latent, dtype=torch.float32)  # (F, 16)
        return item


def compute_action_stats(dataset: BairActionLatentDataset, n_samples: int = 2000, seed: int = 0):
    """Per-dimension mean/std of actions_per_latent over (a subsample of) the
    training set, computed ONCE and cached in the training checkpoint (see
    CLAUDE.md action normalization note). Only nonzero (i>=1) latent frames
    are counted -- latent 0 is always zero by design, would bias std down."""
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(dataset))
    idxs = rng.choice(len(dataset), size=n, replace=False)
    all_actions = []
    for i in idxs:
        a = dataset[int(i)]["actions_per_latent"][1:]  # skip latent 0
        all_actions.append(a.numpy())
    all_actions = np.concatenate(all_actions, axis=0)  # (n*(F-1), 16)
    mean = all_actions.mean(axis=0)
    std = all_actions.std(axis=0) + 1e-6
    return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)
