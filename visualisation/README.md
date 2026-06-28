# Embedding Visualisation (t-SNE / UMAP)

Reproduces the reference figure: 2D projections of model embeddings, colored by
activity (**red = active, blue = inactive**), for each task and each architecture
(CNNModel1/CNNModel2, ViT, YOLOv11).

Two stages, mirroring `shap/`:

1. **`extract_embeddings.py`** — runs a trained `.pth` over a data split and saves
   the **penultimate feature vector** (the input to the final classifier `Linear`,
   captured with a `forward_pre_hook`) per compound → `.npz`.
2. **`plot_projections.py`** — projects those embeddings with t-SNE / UMAP and
   draws per-task panels and multi-task grids.

`run_all.py` chains both over a whole folder of models.

## Why the penultimate layer
The final classifier reads a fixed-width feature vector; that vector is the
learned representation. Projecting it shows whether the model separates
active/inactive — exactly what the reference t-SNE panels visualise. The hook
target per architecture:

| Architecture        | Final Linear (hooked) | Embedding dim |
|---------------------|-----------------------|---------------|
| CNNModel1/CNNModel2 | `fc3`                 | `fc2`         |
| ViT (Swinv2)        | `vit.classifier`      | `hidden_size` |
| YOLOv11             | `model[-1].linear/fc` | head in-feat  |

## Quick start (on the training server)

```bash
# One model:
python visualisation/extract_embeddings.py \
    --model_path trained_models/.../CHEMBL4282-...-CNNModel1-512-256-...-state_dict.pth \
    --task CHEMBL4282 --split test --cuda 0

# Everything under trained_models/, then build grids + panels:
python visualisation/run_all.py --models_dir trained_models --split test --cuda 0
```

Outputs:
- embeddings → `visualisation/embeddings/<task>__<model>__<split>.npz`
- figures    → `visualisation/figures/` (`grid__<model>__<method>.png`, per-task panels)

## Notes
- Architecture and CNN `fc1/fc2` are auto-parsed from the canonical filename;
  override with `--model_name/--fc1/--fc2` if a name is non-standard.
- ViT params are read from `config/config.yaml`.
- t-SNE perplexity / UMAP n_neighbors auto-scale to small test splits.
- UMAP needs `pip install umap-learn`; t-SNE uses scikit-learn (already a dep).
- Embeddings use the **test** split by default (held-out, matches the ROC panels).
