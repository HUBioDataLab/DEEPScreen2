# Monkeypox Use-Case: Viral DNA Polymerase Inhibition

This example demonstrates how to use DEEPScreen2 to train a model for identifying potential inhibitors of the Monkeypox virus DNA polymerase. The dataset includes active and inactive compounds collected from literature and DrugBank.

## Abstract
**AI-driven discovery of antiretroviral drug bictegravir and etravirine as inhibitors against monkeypox and related poxviruses**

Monkeypox virus (MPXV) caused the 2022–2023 global mpox and the concurrent outbreaks in Africa, disproportionately affecting immunocompromised individuals such as people living with HIV. With no approved treatment available, we developed a robust artificial intelligence (AI) pipeline for discovering broad-spectrum poxvirus inhibitors that target the viral DNA polymerases. Among the identified leading candidates, we found that the clinically used antiretroviral drugs bictegravir and etravirine potently inhibit MPXV clade Ia, Ib and IIb infections in human intestinal and skin organoids. The broad anti-poxvirus activities of bictegravir and etravirine were further demonstrated against infections of other Orthopoxviruses such as vaccinia virus and cowpox virus. These findings support the repurposing of bictegravir and etravirine for treating mpox, especially for patients co-infected with HIV, warranting follow-up clinical investigation. The established AI pipeline and our antiviral drug discovery strategies bear major implications for responding to the ongoing mpox emergency and preparing for future poxvirus epidemics.

## Goal
To predict small molecules that can inhibit Monkeypox viral DNA polymerase, aiding in the discovery of potential antiviral treatments.

## Data Source
*   **Activity Data**: `training_files/monkeypox/activity_data.csv` - Contains known active and inactive compounds.
*   **Prediction Data**: `prediction_file/drugbank_mols.csv` - A subset of DrugBank molecules to be screened.

## Usage

Run all commands from the repository root directory (`DEEPScreen2/`).

### 1. Training

Train a model using the provided dataset. We specify the `training_dir` to point to this example's folder.

```bash
python main_training.py \
    --target_id monkeypox \
    --model CNNModel1 \
    --training_dir examples/monkeypox/training_files \
    --pchembl_threshold 5.8 \
    --en monkeypox_experiment \
    --project_name deepscreen_monkeypox
```

### 2. Evaluation

Evaluation results (metrics, confusion matrices) will be automatically saved to:
`result_files/experiments/monkeypox_experiment/`

### 3. Prediction

Use the trained model to predict activity for new molecules (DrugBank dataset).

```bash
python predict_deepscreen.py \
    --target_id monkeypox_prediction \
    --model_path examples/monkeypox/model/monkeypox_best_val-monkeypox-CNNModel1-512-256-0.00001-64-0.2-100-deepscreen_scaffold_balanced_lr0.00001_drop0.2_bs64-state_dict.pth \
    --smiles_file examples/monkeypox/prediction_file/drugbank_mols.csv
```
*(Note: Replace the model path with your newly trained model path if you wish to use your own instead of the pre-trained one provided).*

## Outputs

*   **Trained Model**: Saved in `trained_models/` (or use the one in `examples/monkeypox/model/`).
*   **Predictions**: Saved in `result_files/experiments/`.

## Citation

```bibtex
@article{wang2025ai,
  title={AI-driven discovery of antiretroviral drug bictegravir and etravirine as inhibitors against monkeypox and related poxviruses},
  author={Wang, Yining and {\"U}nl{\"u}, Atabey and Wang, Xin and {\c{C}}evrim, Elif and Offermans, Dewy Mae and Flesseman, Myrthe P and Zaeck, Luca M and Wu, Liping and Bijvelds, Marcel JC and Sam-Agudu, Nadia A and others},
  journal={Communications Biology},
  volume={8},
  number={1},
  pages={1734},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```
