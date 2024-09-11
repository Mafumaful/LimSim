#!/bin/zsh

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

echo "\n# >>>>>> LIMSIM >>>>>>>>>>\nexport LIMSIM_DIR=$SCRIPT_DIR\n# <<<<<< LIMSIM <<<<<<<<\n" >> ~/.zshrc
