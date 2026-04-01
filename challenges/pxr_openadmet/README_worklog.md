# OpenADMET PXR challenge worklog

Workspace: /Users/atabeyunlu/DEEPScreen2/challenges/pxr_openadmet

Initial notes:
- Challenge Space: https://huggingface.co/spaces/openadmet/pxr-challenge
- Dataset: https://huggingface.co/datasets/openadmet/pxr-challenge-train-test
- Activity submission expects exactly 513 rows and columns: SMILES, Molecule Name, pEC50.
- Structure track is separate with 78 blinded fragment-sized molecules.

Goal for first pass:
- Inspect data distributions and duplicates
- Decide how to adapt DEEPScreen2 from binary classification to pEC50 regression
- Build a reproducible baseline and submission file

macOS env notes:
- Apple Silicon machine: M1 Pro
- Working env: /Users/atabeyunlu/DEEPScreen2/.venv-macos311
- Create via: uv venv --python 3.11 .venv-macos311
- Install via: uv pip install --python .venv-macos311/bin/python -r challenges/pxr_openadmet/requirements-macos-py311.txt
- Important: NumPy must stay <2 for RDKit/Chemprop compatibility in this setup.
- CairoSVG import on macOS needs DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/cairo/lib
- Activation helper: challenges/pxr_openadmet/activate_deepscreen_macos.sh
