from pathlib import Path
import torch
import numpy as np
import os
from attention_approximation.pytorch import WORLD_SIZE, LOCAL_RANK, RANK
from attention_approximation.utils import LOGGER

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DistributedDataLoader:
    def __init__(self, path: str | Path, batch_size: int, seq_len: int, split: str):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rank = max(RANK, 0)
        self.num_processes = WORLD_SIZE
        self.current_shard = None

        path = Path(path) if isinstance(path, str) else path
        assert path.exists()
        # get the shard filenames
        shards = sorted((s for s in path.glob("shard_*") if split in s.name), key=lambda x: x.name)
        shards = [s.as_posix() for s in shards]
        assert len(shards) > 0, f"no shards found for split {split}"
        self.shards = shards
        if LOCAL_RANK in {-1, 0}:
            LOGGER.info(f"Found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        if self.current_shard != 0:
            self.current_shard = 0
            self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.batch_size * self.seq_len * self.rank

    def next_batch(self):
        B, T = self.batch_size, self.seq_len
        R, W = self.rank, self.num_processes

        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * W
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * W + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * R
        return x, y

