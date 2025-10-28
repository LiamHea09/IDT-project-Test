# MediaPipe Hand Highlight Demo

This small demo captures your webcam, detects hands with MediaPipe, draws landmarks, and highlights the hand area with a semi-transparent overlay.

Requirements
- Python 3.8+
- Windows PowerShell (example commands below)

Install
```powershell
pip install -r "C:/Users/light/Downloads/Work Expicnece sign language/requirements.txt"
```

Run
```powershell
python "C:/Users/light/Downloads/Work Expicnece sign language/hand_highlight.py"
```

Controls
- Press `q` or `Esc` to quit.

Notes
- If your camera index is not 0, pass `--camera N`.
- You can adjust detection/tracking confidence or model complexity via CLI flags (see `--help`).

Using a convenient `run` command in the venv (Windows)
If you created a venv named `.venv` in the project root, a small helper script is available at
`.venv\Scripts\run.cmd`. After activating the venv, you can start the app by simply typing:

```powershell
# activate the venv first
.\.venv\Scripts\Activate.ps1

# then run the app with no extra typing
run

# pass args as usual, e.g. record label 'A'
run --record A
```

The helper script executes the venv's python and runs `hand_highlight.py`, forwarding any arguments.
If the venv python can't be found the script falls back to the system `python` on PATH.

macOS / Linux
----------------
Virtual environments are not portable between Windows and macOS. To run this project on a Mac or Linux machine, create a fresh venv and use the included mac-friendly helpers:

1. Create and activate a venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Use the project-level script or the venv launcher:

```bash
# run the project-level wrapper (will activate .venv if present)
./run.sh --demo "hello my name judah"

# or use the venv-local launcher created at .venv/bin/run
.venv/bin/run --demo "hello my name judah"
```

Make the scripts executable if needed:

```bash
chmod +x run.sh .venv/bin/run
```
