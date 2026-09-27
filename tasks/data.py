"""Data sources. Every source yields (x, y, mask) batches (see tasks/base.py).

Generator tasks:
    stream  fresh samples from the task generator every step
    file    a pregenerated text file, sampled uniformly
Dataset tasks (fixed data, e.g. MNIST) always sample from the task's own train split.

File format: a JSON header line (`# {...}`) with the generator config, then one example per line as
`split<TAB>input ids<TAB>target ids<TAB>mask`, each field space-separated. Examples are bucketed by length
on load, so batches never mix lengths.
"""
import hashlib
import json
import os

import numpy as np
import torch

FORMAT_VERSION = 2


class StreamSource:
    def __init__(self, task, generator):
        self.task = task
        self.generator = generator

    def batch(self, batch_size):
        return self.task.sample(batch_size, self.generator)


class BucketSource:
    """Samples from pregenerated (x, y, mask) buckets, one bucket per sequence length."""
    def __init__(self, buckets, generator):
        self.buckets = buckets
        self.generator = generator
        # Picking a bucket proportionally to its size then an example uniformly = uniform over examples.
        self.weights = torch.tensor([len(b[0]) for b in buckets], dtype=torch.float)

    def batch(self, batch_size):
        bucket = self.buckets[int(torch.multinomial(self.weights, 1, generator=self.generator))]
        ix = torch.randint(len(bucket[0]), (batch_size,), generator=self.generator)
        return tuple(t[ix] for t in bucket)


class DatasetSource:
    """Samples uniformly (with replacement) from a dataset task's TensorDataset split."""
    def __init__(self, task, dataset, generator):
        self.task = task
        self.dataset = dataset
        self.generator = generator

    def batch(self, batch_size):
        ix = torch.randint(len(self.dataset), (batch_size,), generator=self.generator)
        return self.task.make_batch(*self.dataset[ix])


def write_dataset(task, path, config, num_train, num_val, seed, chunk=1000):
    generator = torch.Generator().manual_seed(seed)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".tmp", "w") as f:
        f.write("# " + json.dumps(config, sort_keys=True) + "\n")
        for split, count in [("train", num_train), ("val", num_val)]:
            for start in range(0, count, chunk):
                x, y, mask = task.sample(min(chunk, count - start), generator)
                for row in zip(x.tolist(), y.tolist(), mask.int().tolist()):
                    f.write(split + "".join("\t" + " ".join(map(str, field)) for field in row) + "\n")
    os.replace(path + ".tmp", path)


def read_dataset(path, config):
    """Return {split: [(x, y, mask), ...]} with one tensor triple per sequence length."""
    groups = {}
    with open(path) as f:
        header = json.loads(f.readline().removeprefix("# "))
        if header != json.loads(json.dumps(config)):
            raise ValueError(f"{path} was generated with {header}, but the config asks for {config}")
        for line in f:
            split, *fields = line.rstrip("\n").split("\t")
            rows = groups.setdefault((split, fields[0].count(" ") + 1), ([], [], []))
            for rows_field, field in zip(rows, fields):
                rows_field.append(field)
    parse = lambda rows, n: torch.from_numpy(np.fromstring(" ".join(rows), dtype=np.int64, sep=" ").reshape(-1, n))
    splits = {"train": [], "val": []}
    for (split, n), (xs, ys, masks) in sorted(groups.items()):
        splits[split].append((parse(xs, n), parse(ys, n), parse(masks, n).bool()))
    return splits


def build_data(task, task_cfg, data_cfg, eval_batch_size, seed, log=print):
    """Return (train source, validation batches, test batches or None). Evaluation sets are fixed."""
    train_gen = torch.Generator().manual_seed(seed)

    def chunks(tensors):
        return [tuple(t[i:i + eval_batch_size] for t in tensors) for i in range(0, len(tensors[0]), eval_batch_size)]

    if task.is_dataset:
        splits = task.load_splits(data_cfg["dir"], data_cfg["num_val"], data_cfg["seed"])
        batches = lambda ds: [task.make_batch(*b) for b in chunks(ds.tensors)]
        return (DatasetSource(task, splits["train"], train_gen), batches(splits["val"]),
                batches(splits["test"]) if "test" in splits else None)

    if data_cfg["mode"] == "stream":
        # The validation set depends only on data.seed so it stays fixed across runs with different seeds.
        val_gen = torch.Generator().manual_seed(data_cfg["seed"])
        n_batches = -(-data_cfg["num_val"] // eval_batch_size)
        return StreamSource(task, train_gen), [task.sample(eval_batch_size, val_gen) for _ in range(n_batches)], None

    if data_cfg["mode"] != "file":
        raise ValueError(f"unknown data mode '{data_cfg['mode']}', expected 'stream' or 'file'")
    config = {"task": task_cfg, "num_train": data_cfg["num_train"], "num_val": data_cfg["num_val"],
              "seed": data_cfg["seed"], "format": FORMAT_VERSION}
    path = data_cfg["path"]
    if path is None:
        digest = hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
        path = os.path.join(data_cfg["dir"], f"{task_cfg['name']}_{digest}.txt")
    if not os.path.exists(path):
        log(f"generating {data_cfg['num_train']:,} train / {data_cfg['num_val']:,} val examples -> {path}")
        write_dataset(task, path, config, data_cfg["num_train"], data_cfg["num_val"], data_cfg["seed"])
    log(f"loading dataset {path}")
    splits = read_dataset(path, config)
    val_batches = [b for bucket in splits["val"] for b in chunks(bucket)]
    return BucketSource(splits["train"], train_gen), val_batches, None
