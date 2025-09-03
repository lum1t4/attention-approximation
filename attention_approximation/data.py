from pathlib import Path
import torch
import numpy as np
import os
from attention_approximation.pytorch import WORLD_SIZE, LOCAL_RANK, RANK


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, path: Path, batch_size: int, seq_len: int, process_rank: int, num_processes: int, split: str):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        shards = sorted((s for s in path.glob("*.npy") if split in s.name), key=lambda x: x.name)
        shards = [s.as_posix() for s in shards]
        print(shards)
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if LOCAL_RANK == 0:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.batch_size * self.seq_len * self.process_rank

    def next_batch(self):
        B, T = self.batch_size, self.seq_len
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y


if __name__ == "__main__":
    print(LOCAL_RANK, WORLD_SIZE)
    dataloader = DataLoaderLite(path=Path("data/edu_fineweb10B"), batch_size=8, seq_len=1024, process_rank=1, num_processes=WORLD_SIZE, split='train')
    x, y = dataloader.next_batch()
    while True:
        x, y = dataloader.next_batch()
        print("Input:", x)
        print("Target:", y)
