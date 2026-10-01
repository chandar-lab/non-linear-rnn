"""Train a recurrent model on a synthetic task.

    uv run train.py configs/nru_copying.json
    uv run train.py configs/nru_copying.json --set train.lr=3e-4 model.memory_size=72
    uv run train.py --resume runs/nru_copying_20260926-120000 [--set train.total_steps=200000]
"""
import argparse
import copy
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from logger import Logger
from models import build_model
from tasks import build_task
from tasks.data import build_data

DEFAULTS = {
    "name": "run",
    "seed": 0,
    "device": "auto",
    "data": {
        "mode": "stream",     # "stream": sample on the fly, "file": pregenerated text file
        "dir": "data",        # pregenerated files and downloaded datasets
        "path": None,         # file mode only; None derives <dir>/<task>_<hash>.txt from the task config
        "num_train": 1000000, # file mode only
        "num_val": 2048,
        "seed": 0,            # seed for the file and the validation set, independent of the run seed
    },
    "train": {
        "batch_size": 32,
        "total_steps": 10000,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "betas": [0.9, 0.999],
        "lr_schedule": "constant",  # "constant" or "cosine"
        "warmup_steps": 0,
        "min_lr_ratio": 0.0,        # cosine floor as a fraction of lr
        "max_grad_norm": None,      # None disables clipping (the norm is still logged)
        "bf16": False,
        "log_interval": 50,
        "print_interval": None,     # print a persistent train line every N steps (multiple of log_interval)
        "eval_interval": 500,
        "eval_batch_size": 256,
        "ckpt_interval": 5000,
        "out_dir": "runs",
    },
    "wandb": {"enabled": False, "project": "non-linear-rnn", "entity": None, "group": None, "tags": []},
}


def merge(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def apply_overrides(cfg, overrides):
    """Apply `a.b.c=value` overrides; values are parsed as JSON when possible, else kept as strings."""
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override '{item}' must look like key=value")
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        *parents, leaf = key.split(".")
        node = cfg
        for p in parents:
            node = node.setdefault(p, {})
        if leaf not in node:
            print(f"warning: override adds new key '{key}'")
        node[leaf] = value
    return cfg


def lr_lambda(cfg):
    t = cfg["train"]
    warmup, total = t["warmup_steps"], t["total_steps"]

    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        if t["lr_schedule"] == "constant":
            return 1.0
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return t["min_lr_ratio"] + (1 - t["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * progress))
    if t["lr_schedule"] not in ("constant", "cosine"):
        raise ValueError(f"unknown lr_schedule '{t['lr_schedule']}'")
    return f


def get_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_loss(logits, y, mask):
    """Cross-entropy over the masked positions."""
    return F.cross_entropy(logits[mask].float(), y[mask])


@torch.no_grad()
def evaluate(model, task, batches, device, autocast, prefix="val"):
    """Average loss and metrics over fixed batches, weighted by batch size."""
    model.eval()
    totals, count, sample = {}, 0, None
    for x, y, mask in batches:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        with autocast():
            logits, _ = model(x)
        metrics = {"loss": compute_loss(logits, y, mask).item(), **task.metrics(logits, y, mask)}
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v * len(x)
        count += len(x)
        if sample is None:
            sample = task.format_sample(x[0].cpu(), y[0].cpu(), mask[0].cpu(), logits[0].float().cpu())
    model.train()
    return {f"{prefix}/{k}": v / count for k, v in totals.items()}, sample


def save_checkpoint(path, **state):
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", help="path to a config json")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    parser.add_argument("--resume", metavar="RUN_DIR", help="resume from a run directory's checkpoint")
    args = parser.parse_args()
    if bool(args.config) == bool(args.resume):
        parser.error("pass exactly one of a config path or --resume RUN_DIR")

    # ---- config and run directory ----
    checkpoint = None
    if args.resume:
        run_dir = args.resume.rstrip("/")
        with open(os.path.join(run_dir, "config.json")) as f:
            cfg = json.load(f)
        checkpoint = torch.load(os.path.join(run_dir, "checkpoint.pt"), map_location="cpu", weights_only=False)
    else:
        with open(args.config) as f:
            cfg = merge(DEFAULTS, json.load(f))
    cfg = apply_overrides(cfg, args.set)
    if not args.resume:
        run_dir = os.path.join(cfg["train"]["out_dir"], f"{cfg['name']}_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    tcfg = cfg["train"]

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_float32_matmul_precision("high")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    from torch._inductor import config as inductor_config
    inductor_config.deterministic = True
    device = get_device(cfg["device"])
    autocast = lambda: torch.autocast(device.type, dtype=torch.bfloat16, enabled=tcfg["bf16"])

    logger = Logger(cfg, os.path.basename(run_dir), run_dir, wandb_id=checkpoint and checkpoint["wandb_id"])

    # ---- task, data, model, optimizer ----
    task = build_task(cfg["task"])
    train_source, val_batches, test_batches = build_data(task, cfg["task"], cfg["data"], tcfg["eval_batch_size"],
                                                         cfg["seed"], logger.print)
    model = build_model(cfg["model"], task.input_size, task.output_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], betas=tuple(tcfg["betas"]),
                                  weight_decay=tcfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(cfg))
    step = 0
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        train_source.generator.set_state(checkpoint["data_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        step = checkpoint["step"]

    info = {
        "parameters": f"{sum(p.numel() for p in model.parameters()):,}",
        "device": str(device),
        "run dir": run_dir,
    }
    if task.baseline_loss is not None:
        info["baseline loss"] = task.baseline_loss
    if step:
        info["resumed at step"] = step
    logger.header(cfg, info)
    logger.samples(val_batches[0])

    def checkpoint_now():
        save_checkpoint(os.path.join(run_dir, "checkpoint.pt"), model=model.state_dict(),
                        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), step=step,
                        data_rng=train_source.generator.get_state(), torch_rng=torch.get_rng_state(),
                        wandb_id=logger.wandb_id)

    # ---- training loop ----
    logger.start(tcfg["total_steps"], step)
    max_norm = tcfg["max_grad_norm"] or float("inf")
    print_interval = tcfg.get("print_interval")
    if print_interval and print_interval % tcfg["log_interval"]:
        raise ValueError("train.print_interval must be a multiple of train.log_interval")
    running_loss, running_steps, tokens, t0 = 0.0, 0, 0, time.perf_counter()
    val_metrics = {}
    model.train()
    try:
        while step < tcfg["total_steps"]:
            x, y, mask = (t.to(device, non_blocking=True) for t in train_source.batch(tcfg["batch_size"]))
            with autocast():
                logits, _ = model(x)
            loss = compute_loss(logits, y, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            scheduler.step()
            step += 1
            running_loss += loss.detach()
            running_steps += 1
            tokens += x.numel()
            logger.step()

            if step % tcfg["log_interval"] == 0:
                avg_loss = running_loss.item() / running_steps
                elapsed = time.perf_counter() - t0
                metrics = {
                    "train/loss": avg_loss,
                    **{f"train/{k}": v for k, v in task.metrics(logits, y, mask).items()},
                    "train/grad_norm": grad_norm.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "perf/steps_per_sec": running_steps / elapsed,
                    "perf/tokens_per_sec": tokens / elapsed,
                }
                if device.type == "cuda":
                    metrics["perf/max_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
                logger.log_train(step, metrics, print_line=bool(print_interval) and step % print_interval == 0)
                running_loss, running_steps, tokens, t0 = 0.0, 0, 0, time.perf_counter()
                if not math.isfinite(avg_loss):
                    logger.print(f"[bold red]non-finite loss at step {step}, stopping")
                    break

            if step % tcfg["eval_interval"] == 0 or step == tcfg["total_steps"]:
                val_metrics, sample = evaluate(model, task, val_batches, device, autocast)
                logger.log_eval(step, val_metrics, sample)
                t0 = time.perf_counter()  # don't count eval time in throughput

            if step % tcfg["ckpt_interval"] == 0:
                checkpoint_now()
    except KeyboardInterrupt:
        logger.print(f"[yellow]interrupted at step {step}, saving checkpoint")

    # ---- wrap up ----
    checkpoint_now()
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    summary = {"step": step, **val_metrics}
    if test_batches:
        test_metrics, sample = evaluate(model, task, test_batches, device, autocast, prefix="test")
        logger.log_eval(step, test_metrics, sample)
        summary.update(test_metrics)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.finish(summary)
    logger.print(f"[bold green]done[/] · {run_dir}")


if __name__ == "__main__":
    main()
