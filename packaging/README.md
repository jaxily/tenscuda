# Packaging

## TensCUDA Monitor (`.deb`)

Builds an Ubuntu/Debian package for the card monitor GUI only (not the full TensCUDA/TT-XLA stack).

```bash
# on Ubuntu 24.04 (e.g. the card host)
bash scripts/build_monitor_deb.sh
# → dist/tenscuda-monitor_0.1.0_all.deb

sudo apt install ./dist/tenscuda-monitor_0.1.0_all.deb
# or: sudo dpkg -i dist/tenscuda-monitor_0.1.0_all.deb
```

Then open **TensCUDA Monitor** from Activities, or run `tenscuda-monitor`.
Terminal UI (SSH-friendly): `tenscuda-monitor --tty`.

Tok/s / model activity requires TensCUDA on the same machine (activity JSONL under `~/.cache/tenscuda/`).
MiniMax video jobs are polled from `MINIMAX_URL` (default `http://127.0.0.1:20000`).
Card temp/power/clock work from sysfs alone.
