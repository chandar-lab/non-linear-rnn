"""CLI (rich) and wandb logging."""
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def fmt(v):
    if isinstance(v, float):
        return f"{v:.2e}" if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e4) else f"{v:.4f}"
    return str(v)


def hms(seconds):
    if seconds is None:
        return "-:--:--"
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


class Logger:
    def __init__(self, cfg, run_name, run_dir, wandb_id=None):
        self.console = Console()
        self.progress = None
        self.wandb = None
        wcfg = cfg["wandb"]
        if wcfg["enabled"]:
            import wandb
            self.wandb = wandb.init(
                project=wcfg["project"], entity=wcfg["entity"], group=wcfg["group"], tags=wcfg["tags"],
                name=run_name, config=cfg, dir=run_dir, id=wandb_id, resume="allow" if wandb_id else None,
            )

    @property
    def wandb_id(self):
        return self.wandb.id if self.wandb else None

    def print(self, *args, **kwargs):
        (self.progress.console if self.progress else self.console).print(*args, **kwargs)

    def header(self, cfg, info):
        self.console.rule(f"[bold cyan]{cfg['name']}")
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="dim")
        table.add_column(overflow="fold")
        for k, v in flatten(cfg).items():
            table.add_row(k, fmt(v))
        table.add_row("", "")
        for k, v in info.items():
            table.add_row(f"[bold]{k}", f"[bold]{fmt(v)}")
        self.console.print(table)
        self.console.rule()

    def samples(self, batch, n=3):
        """Print the first n examples of a batch verbatim as token ids (mask as 0/1)."""
        x, y, mask = batch
        self.console.print(f"[bold]first {min(n, len(x))} validation examples[/] [dim](length {x.shape[1]})")
        for i in range(min(n, len(x))):
            for name, t in (("x", x[i]), ("y", y[i]), ("mask", mask[i].int())):
                prefix = f"#{i}" if name == "x" else ""
                self.console.print(f"[dim]{prefix:>3} {name:<4}[/] {' '.join(map(str, t.tolist()))}", highlight=False)
        self.console.rule()

    def start(self, total_steps, step):
        self.progress = Progress(
            TextColumn("[bold cyan]train"), BarColumn(bar_width=30), MofNCompleteColumn(),
            TimeElapsedColumn(), TextColumn("eta"), TimeRemainingColumn(),
            TextColumn("{task.fields[stats]}"), console=self.console,
        )
        self.progress.start()
        self.bar = self.progress.add_task("train", total=total_steps, completed=step, stats="")

    def step(self):
        self.progress.advance(self.bar)

    def log_train(self, step, metrics, print_line=False):
        short = {k.split("/")[-1]: v for k, v in metrics.items() if not k.startswith("perf/")}
        stats = " · ".join(f"{k} {fmt(v)}" for k, v in short.items())
        stats += f" · {metrics['perf/tokens_per_sec']:,.0f} tok/s"
        self.progress.update(self.bar, stats=stats)
        if print_line:
            # A persistent line (unlike the live bar), so it survives in scrollback and wandb's console log.
            bar = self.progress.tasks[self.bar]
            line = " │ ".join(f"{k} {fmt(v)}" for k, v in short.items())
            perf = f"{metrics['perf/steps_per_sec']:.2f} it/s · {metrics['perf/tokens_per_sec']:,.0f} tok/s"
            if "perf/max_mem_gb" in metrics:
                perf += f" · {metrics['perf/max_mem_gb']:.2f} GB"
            times = f"elapsed {hms(bar.elapsed)} · eta {hms(bar.time_remaining)}"
            self.print(f"[cyan]train[/] step {step:>7}/{bar.total:.0f} │ {line} │ {perf} │ {times}")
        if self.wandb:
            self.wandb.log(metrics, step=step)

    def log_eval(self, step, metrics, sample=None):
        line = " │ ".join(f"{k.split('/')[-1]} [bold]{fmt(v)}[/]" for k, v in metrics.items())
        split = next(iter(metrics)).split("/")[0]  # "val" or "test"
        self.print(f"[magenta]{split:<5}[/] step {step:>7} │ {line}")
        if sample:
            self.print(f"[dim]     sample[/]  {sample}")
        if self.wandb:
            self.wandb.log(metrics, step=step)

    def finish(self, summary):
        if self.progress:
            self.progress.stop()
            self.progress = None
        if self.wandb:
            self.wandb.summary.update(summary)
            self.wandb.finish()
