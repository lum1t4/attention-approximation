"""
This script converts a Hugging Face dataset into a format suitable for training a model.

References:
- https://github.com/karpathy/build-nanogpt/blob/master/fineweb.py
- https://huggingface.co/docs/datasets/overview
-  https://github.com/recursal/RADLADS-paper
"""

from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import os
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
from pathlib import Path
from functools import partial

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


def tokenize_document(doc, tokenizer, token_dtype):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [tokenizer.eos_token_id] # the special <|endoftext|> token delimits all documents
    tokens.extend(tokenizer(doc["text"], add_special_tokens=False)['input_ids'])
    tokens = np.array(tokens, dtype=token_dtype)
    return tokens


def main(
        dataset: str,
        tokenizer: str,
        output_dir: str | Path,
        remote_name: str | None = None,
        shard_size: int = 10**8, # 100M tokens per shard, total of 100 shards
        hf_token: str | None = None
    ):
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(dataset, name=remote_name, split='train', streaming=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False, legacy=False, token=hf_token)
    token_dtype = np.int32 if tokenizer.vocab_size > 2**16 else np.uint16
    nprocs = max(1, os.cpu_count() // 2)
    tokenize_fn = partial(tokenize_document, tokenizer=tokenizer, token_dtype=token_dtype)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=token_dtype)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize_fn, dataset, chunksize=16):
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(output_dir, f"edufineweb_{split}_{shard_index:06d}")
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder

        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(output_dir, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert a Hugging Face dataset into a format suitable for training a model.")
    parser.add_argument('--dataset', type=str, default="HuggingFaceFW/fineweb-edu", help='The name of the Hugging Face dataset to use.')
    parser.add_argument('--remote_name', type=str, required=False, help='The name of the remote dataset to use.') # default="sample-10BT"
    parser.add_argument('--shard_size', type=int, default=10**8, help='The number of tokens per shard.')
    parser.add_argument('--tokenizer', type=str, default="facebook/MobileLLM-350M-layer-share", help='The name of the Hugging Face tokenizer to use.')
    parser.add_argument('--hf-token', type=str, required=False, help='The Hugging Face token for authentication.')
    parser.add_argument('--output-dir', type=str, default="data/edu_fineweb10B", help='The directory to save the output shards.')
    args = parser.parse_args()
    main(**vars(args))
