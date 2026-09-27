# non-linear-rnn

Revisiting non-linear and less common RNN architectures (starting with the Non-saturating Recurrent Unit)
on synthetic long-range memory tasks and psMNIST, alongside GRU and LSTM baselines. Everything is PyTorch,
every task is tokens in / classes out with a loss mask, and every training run is fully described by one
JSON config.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync          # creates .venv with Python 3.12, torch, numpy, rich, wandb
wandb login      # only needed if wandb logging is enabled in the config
```

Run everything through `uv run ...` (or activate `.venv` yourself).

## Quick start

```bash
# train NRU on the copying memory task (configs/nru_<task>.json exist for every task)
uv run train.py configs/nru_copying.json

# same task with another architecture: replace the whole model section
uv run train.py configs/nru_psmnist.json --set 'model={"name":"lstm","embed_size":64,"num_layers":2}'

# same run with some values overridden from the CLI
uv run train.py configs/nru_copying.json --set train.lr=3e-4 task.time_lag_max=200 wandb.enabled=false

# on a Mac / CPU, disable torch.compile
uv run train.py configs/gru_copying.json --set model.compile=false

# resume an interrupted run (optionally extend it)
uv run train.py --resume runs/nru_copying_20260926-120000 --set train.total_steps=200000
```

## Project layout

```
train.py              training entry point
logger.py             rich CLI log + wandb logging
configs/              one JSON per run: task, data, model, train, wandb
models/
  __init__.py         name -> model class registry, build_model()
  nru.py              Non-saturating Recurrent Unit
  gru.py              GRU (reset-after, same equations as torch.nn.GRU)
  lstm.py             LSTM (forget bias init 1)
tasks/
  __init__.py         name -> task class registry, build_task()
  base.py             Task interface (tokens in, classes out, loss mask)
  copying_memory.py   copying memory
  denoising.py        denoising
  adding.py           adding problem
  copy.py             copy
  repeat_copy.py      repeat copy
  associative_recall.py  associative recall
  psmnist.py          permuted sequential MNIST
  data.py             data sources: stream / pregenerated file / fixed dataset
runs/                 run outputs (created on first run, gitignored)
data/                 pregenerated datasets and downloaded MNIST (gitignored)
```

## Features

### Config-driven runs
A config JSON describes the whole run: task, data, model, optimisation and logging. Missing keys fall
back to `DEFAULTS` in `train.py`, and the fully resolved config is saved to the run directory and sent to
wandb, so every run is reproducible from its own `config.json`.

### CLI overrides
`--set key.subkey=value ...` changes any config value without editing files. Values are parsed as JSON
when possible (`3e-4`, `true`, `null`, `[0.9, 0.95]`), otherwise kept as strings. Setting a key that is not
in the config prints a warning, which catches typos. Model and task keys are passed straight to their
constructors, so a misspelled architecture hyperparameter raises an error.

### Models
All three architectures use the same scaffold, so they can be compared fairly:

- token embedding -> stack of recurrent layers -> linear head, one prediction per time step
- pre-norm residual stacking: `u <- u + cell(RMSNorm(u))`, final RMSNorm on the output
- hand-written cells (no cuDNN), run step by step
- optional `torch.compile` over fixed-size chunks of time steps (`chunk_size`). This keeps cross-step
  fusion without the very slow compile of one graph over the whole sequence. It's intended for CUDA.

Each architecture lives in its own self-contained file.

| model  | specific hyperparameters                                  | params (embed 32, 3 layers) |
|--------|-----------------------------------------------------------|-----------------------------|
| `nru`  | `memory_size` (50), `num_heads` (2); `num_heads * memory_size` must be a perfect square | 27.4k |
| `gru`  | `hidden_size` (defaults to `embed_size`)                  | 19.7k |
| `lstm` | `hidden_size` (defaults to `embed_size`), `forget_bias` (1.0) | 25.7k |

Shared model keys: `name`, `embed_size`, `num_layers`, `chunk_size` (16), `compile` (false). When
`hidden_size != embed_size`, GRU/LSTM add a linear projection back into the residual stream.

### Tasks

All tasks follow [myTorch's task folder](https://github.com/apsarath/myTorch/tree/tasks/myTorch/task) and
share one format. Every batch is `(x, y, mask)`, all of shape `(batch, length)`:

- `x` holds input token ids
- `y` holds target class ids, with one prediction per position
- `mask` selects the positions that count towards the loss (cross-entropy) and metrics

Tasks with real-valued or bit-vector inputs in myTorch are tokenised: each bit vector becomes one symbol,
and adding values become digits.

| task | `name` | loss positions | default settings | tokenisation vs. myTorch |
|------|--------|----------------|------------------|--------------------------|
| Copying memory | `copying_memory` | all (or recall only) | `seq_len` 10, `time_lag_min/max` 100, `num_digits` 8, `num_noise_digits` 1, `mask_recall_only` false | already tokens |
| Denoising | `denoising` | last `seq_len` | same as copying memory | already tokens |
| Adding | `adding` | last step | `seq_len` 100, `num_levels` 10 | values are digits in `[0, num_levels)`; the target is their exact sum as a class |
| Copy | `copy` | output half | `num_bits` 8, `min_len` 1, `max_len` 20 | each vector is one of `2^(num_bits-1)` symbols, plus delimiter and blank tokens |
| Repeat copy | `repeat_copy` | output part | `num_bits` 8, `min/max_len` 1–10, `min/max_repeat` 1–10 | as copy, and the repeat count gets its own token instead of a normalised real |
| Associative recall | `associative_recall` | answer + end | `num_bits` 8, `min/max_len` 2–6, `block_len` 3 | each vector is one of `2^(num_bits-2)` symbols, plus item-delimiter, query-marker and blank tokens |
| psMNIST | `psmnist` | last step | `permute` true, `permutation_seed` 0 | pixel intensities 0–255 are tokens (the paper feeds real values) |

**Samples.** Below is one validation example per task, verbatim as generated with the settings of
`configs/nru_<task>.json`: `x` is input tokens, `y` is targets, `mask` marks loss positions. Every run
also prints the first 3 validation examples like this before training starts.

<details>
<summary><b>Copying memory</b> (seq_len=10, time_lag_min=100, time_lag_max=100, num_digits=8, num_noise_digits=1; length 120, vocab 10, 9 classes)</summary>

```
x    8 6 6 6 1 2 6 2 4 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9 0 0 0 0 0 0 0 0 0 0
y    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8 6 6 6 1 2 6 2 4 1
mask 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
```
</details>

<details>
<summary><b>Denoising</b> (seq_len=10, time_lag_min=100, time_lag_max=100, num_digits=8, num_noise_digits=1; length 111, vocab 10, 9 classes)</summary>

```
x    0 0 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6 0 0 0 0 0 1 4 0 0 0 4 0 0 0 0 0 0 4 8 0 0 0 0 2 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9 0 0 0 0 0 0 0 0 0 0
y    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8 6 1 4 4 4 8 2 4 6
mask 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1
```
</details>

<details>
<summary><b>Adding</b> (seq_len=100; length 201, vocab 12, 19 classes)</summary>

```
x    4 9 3 0 3 9 7 3 7 3 1 6 6 9 8 6 6 8 4 3 6 9 1 4 4 1 9 9 9 0 1 2 3 0 5 5 2 9 1 8 8 3 6 9 1 7 3 5 2 1 0 9 3 1 1 0 3 6 6 7 9 6 3 4 5 0 8 2 8 2 7 5 0 0 8 1 9 6 1 0 2 9 4 3 9 3 9 3 9 8 5 3 2 8 5 6 6 5 7 2 10 10 10 10 10 11 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 11 10 10 10 10
y    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 15
mask 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
```
</details>

<details>
<summary><b>Copy</b> (num_bits=8, min_len=1, max_len=20; length 12, vocab 130, 130 classes)</summary>

```
x    47 117 64 67 123 128 129 129 129 129 129 129
y    129 129 129 129 129 129 47 117 64 67 123 128
mask 0 0 0 0 0 0 1 1 1 1 1 1
```
</details>

<details>
<summary><b>Repeat copy</b> (num_bits=8, min_len=1, max_len=10, min_repeat=1, max_repeat=10; length 67, vocab 140, 130 classes)</summary>

```
x    117 64 67 123 67 128 139 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129
y    129 129 129 129 129 129 129 117 64 67 123 67 128 117 64 67 123 67 128 117 64 67 123 67 128 117 64 67 123 67 128 117 64 67 123 67 128 117 64 67 123 67 128 117 64 67 123 67 128 117 64 67 123 67 128 117 64 67 123 67 128 117 64 67 123 67 128
mask 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
```
</details>

<details>
<summary><b>Associative recall</b> (num_bits=8, min_len=2, max_len=6, block_len=3; length 33, vocab 67, 67 classes)</summary>

```
x    47 53 0 64 59 3 39 64 19 21 50 64 23 6 24 64 12 58 1 64 39 23 46 64 65 59 3 39 65 66 66 66 66
y    66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 19 21 50 65
mask 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1
```
</details>

<details>
<summary><b>psMNIST</b> (permute=True, permutation_seed=0; length 784, vocab 256, 10 classes)</summary>

```
x    0 0 0 0 0 0 0 0 0 0 0 0 0 0 240 216 0 0 0 0 0 254 246 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 215 0 0 0 0 0 19 249 0 195 143 0 0 0 0 0 0 86 0 0 0 0 0 0 0 0 0 0 0 0 254 0 178 0 0 0 130 0 0 0 0 0 0 206 0 0 0 0 0 0 0 0 0 0 0 10 0 0 0 39 54 254 0 0 0 0 0 0 177 86 0 65 0 0 0 0 251 0 0 0 0 0 0 0 0 63 0 0 0 0 0 0 24 173 0 0 0 0 15 67 0 0 0 0 0 44 0 0 64 0 0 242 0 0 0 0 0 0 0 58 0 0 0 0 0 13 44 0 0 254 0 142 0 0 0 0 0 0 164 0 0 0 24 37 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 120 0 0 171 0 0 0 0 0 0 0 22 16 0 0 0 0 0 0 0 60 0 0 254 0 0 12 0 0 0 0 0 254 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 24 0 0 0 0 0 187 0 0 0 254 0 0 0 0 0 42 0 0 29 0 254 19 0 0 10 0 0 0 0 149 0 0 0 0 0 0 0 0 0 0 254 4 0 0 0 0 0 0 159 0 0 0 0 0 0 0 254 0 0 0 0 254 0 0 0 0 0 0 0 0 0 0 0 13 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 71 0 0 0 0 0 0 0 0 0 0 0 0 241 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 240 0 0 0 0 0 0 0 0 77 0 0 0 0 0 0 9 0 207 0 0 0 0 0 0 230 0 0 0 20 0 0 0 0 0 241 0 0 0 251 0 0 254 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 252 0 254 0 0 254 135 0 219 0 242 0 0 0 0 0 0 0 0 0 0 0 0 254 0 65 0 183 0 0 0 0 0 0 0 0 0 0 0 148 0 0 0 0 0 0 0 0 0 254 0 0 0 0 254 0 0 0 215 0 254 0 0 0 0 0 0 0 0 249 0 0 9 0 0 0 0 254 0 0 0 0 0 254 0 0 0 0 0 0 0 0 254 0 0 0 0 0 31 0 165 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 247 0 185 247 0 0 0 0 185 0 0 0 0 0 250 0 0 0 0 0 0 0 254 0 0 0 0 0 0 0 0 0 0 135 0 0 84 0 0 0 0 0 0 0 0 254 0 245 0 0 0 0 60 254 0 0 0 0 158 0 0 254 0 254 254 0 0 0 0 0 0 0 0 0 241 0 0 245 0 0 254 0 0 0 0 0 0 254 0 0 0 0 0 0 0 0 0 0 0 226 149 0 0 0 0 0 0 0 0 60 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 254 0 0 254 0 159 0 0 238 0 0 0 0 0 0 0 0 0 0 0 0 0 146 0 0 0 0 0 0 0 116 0 0 58 1 0 0 0 0 0 0 0 0 0 20 0 0 0 0 0 48 0 0 0 0 46 0 0 0 0 0
y    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
mask 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
```
</details>

Each task file's docstring has the exact sequence layout. More details:

- **Copying memory:** the input is `seq_len` digits, noise, a marker at `seq_len + time_lag - 1`, then
  noise. The target is 0 except for the last `seq_len` positions. The loss covers every position as in
  myTorch, or only the recall positions with `mask_recall_only: true`. Metrics are `recall_acc` and `seq_acc`.
- **Denoising:** the digits are scattered at random ordered positions inside the noise, followed by a marker.
  The model outputs the digits in order after the marker.
- **Adding:** as in myTorch, values and markers come one after the other in time, not as two channels.
- **Copy, repeat copy, associative recall:** lengths and repeat counts are sampled once per batch.
  Associative recall samples the query item per example (myTorch uses one per batch).
- **psMNIST:** MNIST is downloaded to `<data.dir>/mnist` on first use. The 60k training images are split
  into train and val (val size `data.num_val`, 5000 in the config), and the official 10k test set is
  evaluated at the end of training. `data.mode` does not apply.
- **Variable lengths:** variable time lags, lengths and repeat counts are sampled once per batch, so a
  batch always shares one length.
- **Metrics:** by default, `acc` (accuracy over masked positions) and `seq_acc` (fraction of sequences with
  every masked position correct).
- **Baseline loss:** where a task defines a trivial reference, the run header shows it (e.g. predict blanks,
  then guess uniformly). A model stuck at that value has learned nothing about memory.

### Data modes

- **`stream`** (default): a fresh batch is generated on the fly every step. The validation set is generated
  once from `data.seed`, so it is identical across runs with different run seeds.
- **`file`**: `num_train + num_val` examples are pregenerated into a text file, then batches are sampled
  from it. The file is created on first use. If `data.path` is `null`, the path is derived from a hash of the
  task and data settings (`<data.dir>/<task>_<hash>.txt`), so different task settings never collide.
  - Format: a JSON header line (`# {...}`) recording the generator settings, then one example per line:
    `split<TAB>input ids<TAB>target ids<TAB>mask`.
  - The header is checked on load, so an existing file generated with different settings raises an error
    instead of being used silently.
  - Examples are bucketed by length, so batches never mix different lengths.
- **Dataset tasks** (psMNIST) ignore `data.mode` and sample from their own fixed train split.
- Batches are sampled uniformly with replacement in every mode, so runs are measured in steps, not epochs.

### CLI training log (rich)

- a header table with the full resolved config, parameter count, device, run directory and baseline loss
- the first 3 validation examples printed verbatim as token ids (`x`, `y`, `mask`), so you can see exactly
  what the model is trained on
- a live progress bar with elapsed time, ETA, and the latest train loss, task metrics, grad norm, LR and
  tokens/sec
- optionally, every `print_interval` steps, a persistent train line with step/total, loss, task metrics,
  grad norm, LR, it/s, tok/s, peak GPU memory (CUDA), elapsed time and ETA. Unlike the live bar, it stays
  in the scrollback and in wandb's captured console log (`output.log`).
- one line per evaluation with validation metrics, plus a sample showing target vs. greedy prediction
  (mismatches in red)

### wandb
Enable it with `wandb.enabled: true`. The run name matches the run directory, the full config is logged,
and `project`, `entity`, `group` and `tags` are configurable. Logged keys:

- `train/loss`, task metrics (e.g. `train/acc`, `train/seq_acc`), `train/grad_norm`, `train/lr`
- `perf/steps_per_sec`, `perf/tokens_per_sec`, `perf/max_mem_gb` (CUDA only)
- `val/loss` and task metrics; `test/...` at the end for tasks with a test split

Resuming a run continues the same wandb run.

### Optimisation
- AdamW with configurable `lr`, `betas` and `weight_decay`
- LR schedule: `constant` or `cosine`, both with optional linear warmup (`warmup_steps`); cosine decays to
  `min_lr_ratio * lr`
- optional global grad-norm clipping (`max_grad_norm`). The grad norm is logged either way, which helps
  diagnose RNN instability.
- optional bf16 autocast (`bf16: true`); the loss is always computed in fp32
- training stops early with a message if the loss becomes NaN/inf

### Run outputs, checkpointing and resume
Each run writes to `runs/<name>_<YYYYmmdd-HHMMSS>/`:

| file | contents |
|------|----------|
| `config.json` | fully resolved config |
| `checkpoint.pt` | model, optimizer, scheduler, step, data-sampler RNG, torch RNG, wandb run id |
| `model.pt` | final model `state_dict` |
| `metrics.json` | final step, validation metrics, and test metrics for tasks with a test split |

A checkpoint is saved every `ckpt_interval` steps, at the end, and on Ctrl-C. `--resume RUN_DIR` reloads
the run's own `config.json` (with any `--set` overrides applied on top), restores all state, and continues
from the saved step. For example, extend a finished run with `--set train.total_steps=...`.

## Config reference

Keys under `task` and `model` depend on the chosen task and architecture (see above). Everything else,
with defaults:

| key | default | notes |
|-----|---------|-------|
| `name` | `"run"` | run name, prefix of the run directory |
| `seed` | 0 | run seed (init, training batches) |
| `device` | `"auto"` | `auto` picks cuda > mps > cpu, or give e.g. `"cuda:1"` |
| `data.mode` | `"stream"` | `stream` or `file` (generator tasks only) |
| `data.dir` | `"data"` | pregenerated files and downloaded datasets |
| `data.path` | `null` | file mode only; `null` derives a path from the settings |
| `data.num_train` | 1000000 | file mode only |
| `data.num_val` | 2048 | validation set size |
| `data.seed` | 0 | seed for the file and the validation set |
| `train.batch_size` | 32 | |
| `train.total_steps` | 10000 | |
| `train.lr` | 1e-3 | |
| `train.weight_decay` | 0.0 | |
| `train.betas` | [0.9, 0.999] | |
| `train.lr_schedule` | `"constant"` | `constant` or `cosine` |
| `train.warmup_steps` | 0 | |
| `train.min_lr_ratio` | 0.0 | cosine floor |
| `train.max_grad_norm` | `null` | `null` disables clipping |
| `train.bf16` | false | |
| `train.log_interval` | 50 | steps between train metric updates (progress bar + wandb) |
| `train.print_interval` | `null` | steps between persistent train lines; must be a multiple of `log_interval`, `null` disables it |
| `train.eval_interval` | 500 | steps between evaluations (also runs at the last step) |
| `train.eval_batch_size` | 256 | |
| `train.ckpt_interval` | 5000 | |
| `train.out_dir` | `"runs"` | |
| `wandb.enabled` | false | the provided configs set this to true |
| `wandb.project` | `"non-linear-rnn"` | |
| `wandb.entity` | `null` | |
| `wandb.group` | `null` | |
| `wandb.tags` | [] | |

## Extending

**New architecture**: add `models/<name>.py` with a model class taking
`(vocab_size, output_size, **model_cfg)`, whose `forward(idx, state=None)` returns `(logits, state)` with
logits of shape `(batch, length, output_size)`. Register it in `MODELS` in `models/__init__.py`.

**New task**: subclass `tasks.base.Task` and set `input_size` (vocab size) and `output_size` (number of
classes). Then implement one of:

- **Generator task:** `sample(batch_size, generator)` returning `(x, y, mask)`. `x` and `y` are
  LongTensors and `mask` a BoolTensor, all of shape `(batch, length)`. Both data modes then work
  automatically.
- **Dataset task:** set `is_dataset = True` and implement `load_splits(data_dir, num_val, seed)`,
  returning TensorDatasets for `train`, `val` and optionally `test`. Also implement `make_batch(*tensors)`,
  which turns a slice of a split into `(x, y, mask)`.

Optionally override `metrics`, `format_sample` and `baseline_loss`. Register the task in `TASKS` in
`tasks/__init__.py`.

## Notes

- `torch.compile` builds one graph per distinct `(batch size, chunk length)`. With variable time lags, the
  last partial chunk varies, so expect a few extra compiles early on (dynamo's cache limit is raised to 64).
  Compiling on CPU is very slow; use `model.compile=false` off-GPU.
- `nru_mods_copying_memory_optimized.ipynb` is the original preliminary notebook this codebase was built from.
