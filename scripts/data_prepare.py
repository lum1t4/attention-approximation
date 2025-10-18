"""
This script converts a Hugging Face dataset into a format suitable for training a model.
WARNING: If dataset or tokenizer are gated behind authentication, it may be necessary to set the HF_TOKEN environment variable.
References:
- https://github.com/karpathy/build-nanogpt/blob/master/fineweb.py
- https://huggingface.co/docs/datasets/overview
- https://github.com/recursal/RADLADS-paper
- https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/indexed_dataset.py
"""


from io import BufferedWriter
from typing import Literal
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import os
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
from functools import partial
import struct

from attention_approximation.data import TokenDatasetIndex

document_lens = []
token_count = 0

COLUMN_NAME = "text"
VERSION = 1


def main(
        dataset: str,
        tokenizer: str,
        output_dir: str | Path,
        name: str | None = None,
        split: str = "train",
    ):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(dataset, name=name, split=split, streaming=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False, legacy=False)
    token_dtype = np.uint32 if tokenizer.vocab_size > 2**16 else np.uint16

    with open(output_dir / f'dataset-{split}.bin', "wb") as index_fd:
        def process(example):
            global token_count, COLUMN_NAME
            batch_tokens = []
            batch_lens = []

            for encoded in tokenizer(example[COLUMN_NAME], add_special_tokens=False)['input_ids']:
                # add the end of text token, e.g. 50256 for gpt2 bpe
                encoded.append(tokenizer.eos_token_id)
                encoded = np.asarray(encoded, dtype=token_dtype)
                batch_tokens.append(encoded)
                batch_lens.append(len(encoded))
                document_lens.append(len(encoded))
                token_count += len(encoded)

            index_fd.write(np.concatenate(batch_tokens).tobytes(order="C"))
            # a return is mandatory to use HF dataset map function
            # TODO: check if there a faster method to achive same result
            return {'ids': batch_tokens, 'len': batch_lens}

        tokenized = ds.map(process, remove_columns=[COLUMN_NAME], batched=True)
        progress_bar = tqdm(tokenized, unit='docs', desc='writing')
        for _ in progress_bar:
            progress_bar.set_description(f'{token_count} tokens')

    lenghts = np.array(document_lens, dtype=np.int32)
    token_size = np.dtype(token_dtype).itemsize # how many bytes to store a token
    offsets = np.cumsum([0, *lenghts[:-1].tolist()], dtype=np.int64) * token_size
    TokenDatasetIndex(
        token_count=sum(lenghts),
        token_byte_size=token_size,
        document_count=lenghts.size,
        document_lenghts=lenghts,
        document_offsets=offsets
    ).save(output_dir / f'index-{split}.bin')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare a HF Dataset to binary files to speed data loading during training")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--name', type=str, required=False, help="Subset name, if the dataset has multiple subsets.")
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--tokenizer', type=str, default="facebook/MobileLLM-350M-layer-share")
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
