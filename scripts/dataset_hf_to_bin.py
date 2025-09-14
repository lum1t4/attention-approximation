"""
This script converts a Hugging Face dataset into a format suitable for training a model.
WARNING: If dataset or tokenizer are gated behind authentication, it may be necessary to set the HF_TOKEN environment variable.
References:
- https://github.com/karpathy/build-nanogpt/blob/master/fineweb.py
- https://huggingface.co/docs/datasets/overview
-  https://github.com/recursal/RADLADS-paper
"""


from typing import Literal
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import os
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
from functools import partial


def tokenize_document(doc: dict, tokenizer, token_dtype: np.uint16 | np.uint32):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [tokenizer.eos_token_id] # the special <|endoftext|> token delimits all documents
    tokens.extend(tokenizer(doc["text"], add_special_tokens=False)['input_ids'])
    tokens = np.array(tokens, dtype=token_dtype)
    return tokens



def main(
        dataset: str,
        tokenizer: str,
        output_dir: str | Path,
        name: str | None = None,
        split: str = "train",
        shard_size: int = 10**8,
        reserve_val_shard: bool = False, # if True, reserves the first shard as a validation shard
    ):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(dataset, name=name, split=split)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False, legacy=False)
    token_dtype = np.uint32 if tokenizer.vocab_size > 2**16 else np.uint16
    nprocs = max(1, os.cpu_count() - 2) # don't hog the entire system
    tokenize = partial(tokenize_document, tokenizer=tokenizer, token_dtype=token_dtype)

    with mp.Pool(nprocs) as pool:
        # preallocate buffer to hold current shard
        shard_index = 0
        token_count = 0
        progress_bar = None
        shard = np.empty((shard_size,), dtype=token_dtype)

        for tokens in pool.imap(tokenize, dataset, chunksize=16):
            # is there enough space in the current shard for the new tokens?
            num_tokens = len(tokens)
            if token_count + num_tokens < shard_size:
                shard[token_count:token_count+num_tokens] = tokens
                token_count += num_tokens
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(num_tokens)
            else:
                # finish current shard
                split = "val" if shard_index == 0 and reserve_val_shard else split
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                shard[token_count:token_count+remainder] = tokens[:remainder]
                np.save(output_dir / f"shard_{split}_{shard_index:06d}", shard)
                shard_index += 1
                progress_bar.close()
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                shard[0:num_tokens - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

    # write last incomplete shard
    if token_count > 0:
        split = "val" if shard_index == 0 and reserve_val_shard else split
        np.save(output_dir / f"shard_{split}_{shard_index:06d}", shard[:token_count])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert a Hugging Face dataset into shard .npy files.")
    parser.add_argument('--dataset', type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument('--name', type=str, required=False, help="Subset name, if the dataset has multiple subsets.")
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--shard_size', type=int, default=10**8)
    parser.add_argument('--reserve_val_shard', action='store_true', help="If set, reserves the first shard as a validation shard.")
    parser.add_argument('--tokenizer', type=str, default="facebook/MobileLLM-350M-layer-share")
    parser.add_argument('--output-dir', type=str, default="data/edu_fineweb10B")
    args = parser.parse_args()
    main(**vars(args))
