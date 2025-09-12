from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import os
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
from functools import partial

def tokenize_document(doc, tokenizer, token_dtype):
    """Tokenizes a single document and returns a numpy array of tokens."""
    tokens = [tokenizer.eos_token_id]
    tokens.extend(tokenizer(doc["text"], add_special_tokens=False)['input_ids'])
    return np.array(tokens, dtype=token_dtype)


def main(
        dataset: str,
        tokenizer: str,
        output_dir: str | Path,
        remote_name: str | None = None,
        shard_size: int = 10**8,
        hf_token: str | None = None,
    ):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset, name=remote_name, split='train', streaming=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False, legacy=False)

    token_dtype = np.int32 if tokenizer.vocab_size > 2**16 else np.uint16
    nprocs = max(1, os.cpu_count() // 2)

    tokenize_fn = partial(tokenize_document, tokenizer=tokenizer, token_dtype=token_dtype)

    shard_index = 0
    all_tokens_np = np.empty((shard_size,), dtype=token_dtype)
    token_count = 0
    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")

    with mp.Pool(nprocs) as pool:
        for tokens in pool.imap(tokenize_fn, dataset, chunksize=16):
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                progress_bar.update(len(tokens))
            else:
                # finish current shard
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(output_dir, f"edufineweb_{split}_{shard_index:06d}")
                remainder = shard_size - token_count
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                np.save(filename, all_tokens_np)

                shard_index += 1
                progress_bar.close()
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")

                leftover = tokens[remainder:]
                all_tokens_np[0:len(leftover)] = leftover
                token_count = len(leftover)
                progress_bar.update(len(leftover))

    # write last incomplete shard
    if token_count > 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(output_dir, f"edufineweb_{split}_{shard_index:06d}")
        np.save(filename, all_tokens_np[:token_count])
    progress_bar.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert a Hugging Face dataset into shard .npy files.")
    parser.add_argument('--dataset', type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument('--remote_name', type=str, required=False)
    parser.add_argument('--shard_size', type=int, default=10**8)
    parser.add_argument('--tokenizer', type=str, default="facebook/MobileLLM-350M-layer-share")
    parser.add_argument('--output-dir', type=str, default="data/edu_fineweb10B")
    args = parser.parse_args()
    main(**vars(args))
