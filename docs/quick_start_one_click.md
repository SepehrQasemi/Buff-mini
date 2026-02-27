# Quick Start (One-Click Launcher)

This guide lets you start Buff-mini UI without typing CLI commands.

## Windows (double-click)

1. Open the repository folder.
2. Double-click `start_buffmini.bat`.
3. The launcher will:
   - create `.venv/` if missing
   - install/update Python dependencies
   - start Streamlit
   - open your browser automatically

## macOS / Linux

Run from the repository root:

```bash
chmod +x start_buffmini.sh
./start_buffmini.sh
```

## What the launcher prints

- repository root path
- venv path
- dependency install status
- local URL (for example `http://localhost:8501`)

If port `8501` is already in use, the launcher will show that URL and start a new instance on the next free port.

## Troubleshooting

- Python missing:
  Install Python 3.11+ and retry.
- Dependency install errors:
  Check internet connectivity and local Python permissions, then re-run launcher.
- Browser did not open:
  Open the printed local URL manually.
