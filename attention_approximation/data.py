import os
import struct
from dataclasses import dataclass
from io import BufferedReader, BufferedWriter
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from attention_approximation.pytorch import RANK, DistributedEvalSampler, seed_worker

"""
WARNING: while taking as reference the followings:
- https://github.com/recursal/RADLADS-paper
- https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/indexed_dataset.py
binaries were not thought to be compatible either the compability has been verified.
It was prefered to keep things simple and understandable at glace.
"""

INDEX_HEADER = b"MMIDIDX\x00\x00"
INDEX_VERSION = 1


@dataclass
class TokenDatasetIndex:
    token_count: int
    token_byte_size: int
    document_count: int
    document_lenghts: np.ndarray
    document_offsets: np.ndarray
    version: int = INDEX_VERSION

    @property
    def token_dtype(self):
        return np.uint16 if self.token_byte_size == 2 else np.uint32

    def save(self, file: BufferedWriter | str | Path):
        fd = file if isinstance(file, BufferedWriter) else open(file, 'wb')
        fd.write(INDEX_HEADER)
        fd.write(struct.pack("<Q", INDEX_VERSION))
        fd.write(struct.pack("<B", self.token_byte_size))
        fd.write(struct.pack("<Q", self.document_count))
        fd.write(self.document_lenghts.tobytes(order="C"))
        fd.write(self.document_offsets.tobytes(order="C"))
        # if the file descriptor is passed, let the user decide when and where close it
        if not isinstance(file, BufferedWriter):
            fd.close()
        return

    @staticmethod
    def load(file: BufferedReader | str | Path):
        fd = file if isinstance(file, BufferedReader) else open(file, 'rb')
        assert fd.read(9) == INDEX_HEADER, "Header mismatch"
        version,  = struct.unpack("<Q", fd.read(8)) # index version
        byte_size, = struct.unpack("<B", fd.read(1)) # token byte size
        count, = struct.unpack("<Q", fd.read(8)) # document count
        lenghts = np.frombuffer(fd.read(count * 4), dtype=np.int32) # token per document
        offsets = np.frombuffer(fd.read(count * 8), dtype=np.int64) # pointers to documents
        # if the file descriptor is passed, let the user decide when and where close it
        if not isinstance(file, BufferedWriter):
            fd.close()
        return TokenDatasetIndex(
            version=version,
            token_count=sum(lenghts),
            token_byte_size=byte_size,
            document_count=count,
            document_lenghts=lenghts,
            document_offsets=offsets
        )


class TokenDataset(torch.utils.data.Dataset):
    index: TokenDatasetIndex

    def __init__(self, path: str | Path, seq_len: int = 512, split: str = "train"):
        super().__init__()
        path = Path(path) if isinstance(path, str) else path
        index_path = path / f'index-{split}.bin'
        token_path = path / f'dataset-{split}.bin'
        assert index_path.exists(), "Could not find index"
        assert token_path.exists(), "Could not find token corpus"

        self.seq_len = seq_len
        self.index = TokenDatasetIndex.load(index_path)
        self.bucket = np.memmap(token_path, dtype=self.index.token_dtype, mode="r")

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        end = start + self.seq_len
        # Load the token sequence
        tokens = torch.tensor(self.bucket[start:end + 1], dtype=torch.long)
        # Ensure we have the full sequence length
        if len(tokens) < self.seq_len + 1:
            # Pad with zeros if needed (though this shouldn't happen with proper dataset sizing)
            tokens = torch.nn.functional.pad(tokens, (0, self.seq_len + 1 - len(tokens)), value=0)
        return tokens[:-1], tokens[1:]

    def __len__(self):
        return self.index.token_count // self.seq_len


def distributed_dataloader(
    dataset: Dataset,
    batch: int = 16,
    workers: int = 8,
    shuffle: bool = True,
    pin_memory: bool = True,
    mode: Literal["train", "valid", "test"] = "train",
) -> DataLoader:
    bs = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), bs if bs > 1 else 0, workers])  # number of workers

    distributed = DistributedSampler if mode == "train" else DistributedEvalSampler
    sampler = distributed(dataset, shuffle=shuffle) if RANK != -1 else None
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=True,  # Drop incomplete batches to prevent shape issues
    )
