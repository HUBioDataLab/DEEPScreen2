# SHAP Vis — Usage

End-to-end pipeline that turns SHAP `.npz` outputs into per-molecule attention
PNGs, packed into one ZIP per model. `app8.py` is the interactive Streamlit
version of the same flow.

## Pipeline overview

```
prepare_csv.py  ->  molecules.csv      (fetch SMILES from ChEMBL / PubChem)
generate_images.py  ->  mol_images/    (transparent 300x300 molecule PNGs)
shap_cli.py     ->  output_zips/<model>.zip   (terminal version of app7/app8 "Download ZIP")
```

## Requirements

```bash
pip install numpy opencv-python matplotlib pillow rdkit pandas requests \
            chembl_webresource_client streamlit
```

## 1. Build `molecules.csv`

Scans `shap_numpy_data/` for `*_0.npz`, collects unique molecule IDs, and looks
up each SMILES (ChEMBL first, PubChem fallback).

```bash
python prepare_csv.py          # writes ./molecules.csv
```

CSV columns: `molecule_id, smiles`.

## 2. Generate molecule images

Renders transparent-background PNGs from the SMILES list.

```bash
python generate_images.py --input molecules.csv --output_dir ./mol_images
```

Missing PNGs are also generated on demand by `shap_cli.py`, so this step is
optional if `molecules.csv` exists.

## 3. Render attention ZIPs

Provide either a single model folder or a parent folder of model subfolders.

```bash
# single model folder -> CHEMBL301_cnn.zip
python shap_cli.py -i shap_numpy_data/CHEMBL301_cnn -o ./output_zips

# parent folder -> one ZIP per subfolder
python shap_cli.py -i shap_numpy_data -o ./output_zips
```

Key options (defaults match the app7/app8 sliders):

| Flag | Default | Meaning |
|------|---------|---------|
| `--input`, `-i` | — | NPZ folder or parent of model subfolders (required) |
| `--output_dir`, `-o` | `./output_zips` | Where ZIPs are written |
| `--mol_images_dir` | `./mol_images` | Transparent molecule PNGs |
| `--molecules_csv` | `./molecules.csv` | SMILES source for missing PNGs |
| `--hotspot_p` | `95.0` | Focus threshold (Top %) |
| `--blur_sigma` | `5.7` | Smoothing |
| `--gamma` | `1.50` | Intensity |
| `--alpha` | `0.75` | Opacity |

## Run everything at once

```bash
./run_pipeline.sh                              # defaults
./run_pipeline.sh --input shap_numpy_data
./run_pipeline.sh --skip-csv --skip-images     # only re-render the ZIPs
```

## Interactive version

```bash
streamlit run app8.py
```

Same parameters exposed as sliders, with a "Download ZIP" button equivalent to
`shap_cli.py`.
