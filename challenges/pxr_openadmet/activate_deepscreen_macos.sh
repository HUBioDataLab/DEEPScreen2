#!/usr/bin/env bash
set -e
cd /Users/atabeyunlu/DEEPScreen2
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/cairo/lib${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}
source .venv-macos311/bin/activate
export PYTHONPATH=/Users/atabeyunlu/DEEPScreen2${PYTHONPATH:+:$PYTHONPATH}
echo "Activated DEEPScreen2 macOS env (.venv-macos311)"
echo "Python: $(python --version)"
