from attention_approximation.data import DistributedDataLoader
from pathlib import Path
from transformers import AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained("facebook/MobileLLM-350M-layer-share", use_fast=False, legacy=False)
dataloader = DistributedDataLoader(
    path=Path("data/minipile"),
    batch_size=8,
    seq_len=12,
    split='train'
)


x, y = dataloader.next_batch()
print(x.shape, y.shape)

bs = x.size(0)
for i in range(bs):
    print(json.dumps({"input": tokenizer.decode(x[i]), "output": tokenizer.decode(y[i])}))



