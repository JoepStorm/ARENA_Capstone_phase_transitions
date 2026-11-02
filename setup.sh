#!/bin/bash
# Sets up the capstone env on top of the ARENA env (ARENA_materials/.venv).
set -e

# LaTeX for matplotlib's usetex (plots use rc('text', usetex=True))
apt-get update
apt-get install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended cm-super dvipng ghostscript

ARENA_VENV="${ARENA_VENV:-$HOME/ARENA_materials/.venv}"
source "$ARENA_VENV/bin/activate"
pip install -r "$(dirname "$0")/requirements.txt"
