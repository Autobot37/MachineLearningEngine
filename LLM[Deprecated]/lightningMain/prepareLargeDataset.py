from multiprocessing import cpu_count
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, DatasetDict

num_proc = cpu_count()//2

num_shards = {"train":32,"val":8}

dataset = load_dataset("openwebtext")

split_dataset = dataset["train"].train_test_split(test_size=0.02,seed=2357,shuffle=True)

shard_dataset = DatasetDict()

for split, dset in split_dataset.items():
    for i in range(num_shards[split]):
        shard_dataset[f"{split}_{i:02}"] = dset.shard(num_shards[split], i)


def process(example):
    ids = tokenizer.encode(example["text"])
    ids.append(tokenizer.eos_id())
    out = {"ids":ids}
    return out

dataset = shard_dataset.map(
    process,
    remove_columns=["text"],
    desc = "mapping",
    num_proc=num_proc
)

#write to file.