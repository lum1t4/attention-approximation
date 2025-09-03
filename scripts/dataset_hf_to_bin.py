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


DATASET_NAME = "HuggingFaceFW/fineweb-edu"
SHARD_SIZE = int(1e8) # 100M tokens per shard, total of 100 shards
TOKENIZER_NAME = "facebook/MobileLLM-350M-layer-share"

# create the cache the local directory if it doesn't exist yet
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

dataset = load_dataset(DATASET_NAME, name=remote_name, split='train', streaming=True)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=False, legacy=False)
token_dtype = np.int32 if tokenizer.vocab_size > 2**16 else np.uint16



def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [tokenizer.eos_token_id] # the special <|endoftext|> token delimits all documents
    tokens.extend(tokenizer(doc["text"], add_special_tokens=False)['input_ids'])
    tokens = np.array(tokens, dtype=token_dtype)
    return tokens


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


def main():
    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((SHARD_SIZE,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize, dataset, chunksize=16):
            if token_count + len(tokens) < SHARD_SIZE:
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=SHARD_SIZE, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                remainder = SHARD_SIZE - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder

        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])


if __name__ == "__main__":
    main()
