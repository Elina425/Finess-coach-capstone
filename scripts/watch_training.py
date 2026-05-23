#!/usr/bin/env python3
"""Live training-progress dashboard for train_xlstm_egoexo_multitask.py.

Reads the run's training_history.json (written each epoch) and prints a
refreshing summary table with ETA. Supports fixed-, phased-, or DWA-weighted runs.

Usage:
    ./venv/bin/python scripts/watch_training.py results/xlstm_egoexo_multitask_uncw
    ./venv/bin/python scripts/watch_training.py results/xlstm_egoexo_multitask_uncw --refresh 10

Stop with Ctrl+C — does not affect the training process.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ANSI_CLEAR = "\033[2J\033[H"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir", type=Path,
                   help="Output dir passed to --output-dir (e.g. results/xlstm_egoexo_multitask_uncw)")
    p.add_argument("--refresh", type=float, default=10.0,
                   help="Seconds between refreshes (default 10).")
    p.add_argument("--log", type=Path, default=None,
                   help="Optional path to the .log file for last raw lines. Auto-detected if not given.")
    return p.parse_args()


def fmt_dt(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def load_history(history_path: Path):
    if not history_path.is_file():
        return None
    try:
        return json.loads(history_path.read_text())
    except Exception:
        return None


def find_log_file(run_dir: Path, explicit: Path | None) -> Path | None:
    if explicit and explicit.is_file():
        return explicit
    # Common conventions used by the run commands in this repo
    parent = run_dir.parent
    name = run_dir.name
    candidates = [
        parent / f"{name}.log",
        run_dir / "train.log",
        run_dir / f"{name}.log",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def find_pid(run_dir: Path) -> int | None:
    pid_file = run_dir.with_suffix(".pid") if run_dir.suffix else run_dir.parent / f"{run_dir.name}.pid"
    if pid_file.is_file():
        try:
            return int(pid_file.read_text().strip())
        except Exception:
            return None
    return None


def process_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False
    return True


def tail_lines(path: Path | None, n: int = 6) -> list[str]:
    if not path or not path.is_file():
        return []
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            chunk = min(size, 8192)
            f.seek(size - chunk)
            data = f.read().decode(errors="replace")
        lines = [ln for ln in data.splitlines() if ln.strip()]
        return lines[-n:]
    except Exception:
        return []


def fmt_row(row: dict) -> str:
    ep = int(row.get("epoch", 0))
    loss = row.get("train_loss", float("nan"))
    cls = row.get("train_cls", float("nan"))
    reg = row.get("train_reg", float("nan"))
    cmt = row.get("train_cmt", float("nan"))
    vacc = row.get("val_accuracy", float("nan"))
    vf1 = row.get("val_f1_macro", float("nan"))
    vmae = row.get("val_mae", float("nan"))
    lr = row.get("lr", float("nan"))
    s = f"  ep {ep:>3d}  loss={loss:6.3f}  cls={cls:5.3f}  reg={reg:5.3f}  cmt={cmt:5.3f}  " \
        f"val_acc={vacc:.4f}  val_f1={vf1:.4f}  val_mae={vmae:.4f}  lr={lr:.2e}"
    sig = row.get("sigmas")
    if sig:
        s += "  σ: " + " ".join(f"{k}={v:.2f}" for k, v in sig.items())
    return s


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    history_path = run_dir / "training_history.json"
    log_path = find_log_file(run_dir, args.log)
    pid = find_pid(run_dir)

    start_time = time.time()
    last_seen_epoch = 0
    last_seen_time = time.time()

    def on_sigint(_signum, _frame):
        print("\n[watch] stopped. Training continues in the background.")
        sys.exit(0)
    signal.signal(signal.SIGINT, on_sigint)

    while True:
        print(ANSI_CLEAR, end="")
        now = datetime.now().strftime("%H:%M:%S")
        alive = process_alive(pid) if pid else None
        alive_s = "🟢 running" if alive else ("🔴 not found" if pid else "❓ pid unknown")

        print(f"┌─ watch_training · {now} · {alive_s}" + (f" (pid {pid})" if pid else ""))
        print(f"│  run_dir : {run_dir}")
        if log_path:
            print(f"│  log     : {log_path}")
        print("│")

        history = load_history(history_path)
        if not history:
            print("│  waiting for training_history.json to appear …")
            print("│  (first epoch usually writes within 1–3 minutes)")
            tail = tail_lines(log_path, n=8)
            if tail:
                print("│")
                print("│  --- log tail ---")
                for ln in tail:
                    print(f"│  {ln[:200]}")
            print("└─")
            time.sleep(args.refresh)
            continue

        n_epochs = len(history)
        last = history[-1]
        # Best so far
        best_by_acc = max(history, key=lambda r: r.get("val_accuracy", -1))
        best_by_f1 = max(history, key=lambda r: r.get("val_f1_macro", -1))
        best_by_mae = min(history, key=lambda r: r.get("val_mae", float("inf")))

        # Progress + ETA
        cur_ep = int(last.get("epoch", n_epochs))
        # try to infer total epochs from --epochs default 50 (best-effort)
        total_eps = 50
        if last_seen_epoch != cur_ep:
            last_seen_time = time.time()
            last_seen_epoch = cur_ep
        elapsed_since_first = time.time() - start_time
        secs_per_epoch = elapsed_since_first / max(1, cur_ep)
        eta_secs = secs_per_epoch * max(0, total_eps - cur_ep)

        bar_w = 36
        frac = min(1.0, cur_ep / total_eps)
        fill = int(bar_w * frac)
        bar = "█" * fill + "░" * (bar_w - fill)
        print(f"│  Progress  [{bar}] {cur_ep}/{total_eps}  ({100*frac:.0f}%)")
        print(f"│  ETA       ~{fmt_dt(eta_secs)}  ({fmt_dt(secs_per_epoch)}/epoch · est. at {(datetime.now()+timedelta(seconds=eta_secs)).strftime('%H:%M:%S')})")
        print("│")

        # Best so far
        print("│  ┌─ best so far ─────────────────────────────────────")
        print(f"│  │ val_acc  {best_by_acc.get('val_accuracy', float('nan')):.4f}  at epoch {int(best_by_acc.get('epoch', 0))}")
        print(f"│  │ val_f1   {best_by_f1.get('val_f1_macro', float('nan')):.4f}  at epoch {int(best_by_f1.get('epoch', 0))}")
        print(f"│  │ val_mae  {best_by_mae.get('val_mae', float('nan')):.4f}  at epoch {int(best_by_mae.get('epoch', 0))}")
        print("│  └────────────────────────────────────────────────────")
        print("│")

        # Last few epochs
        print("│  recent epochs:")
        for row in history[-6:]:
            print(f"│{fmt_row(row)}")

        # Log tail (raw)
        tail = tail_lines(log_path, n=4)
        if tail:
            print("│")
            print("│  --- log tail ---")
            for ln in tail:
                print(f"│  {ln[:200]}")

        print("└─")
        try:
            time.sleep(args.refresh)
        except KeyboardInterrupt:
            on_sigint(None, None)


if __name__ == "__main__":
    sys.exit(main())
