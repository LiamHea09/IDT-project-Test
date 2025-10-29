#!/bin/sh
# Project-level launcher that activates the .venv if present, then runs hand_highlight.py

# Prefer local venv python if available
if [ -x ".venv/bin/python" ]; then
  . .venv/bin/activate
  exec python hand_highlight.py "$@"
else
  exec python3 hand_highlight.py "$@"
fi
