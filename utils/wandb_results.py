import wandb
import pandas as pd
import argparse
from typing import Optional

def get_best_wandb_runs(entity: str, project: str, group_name: Optional[str] = None, 
                        val_metric: str = "Validation/ROC AUC", 
                        test_metric: str = "Test/ROC AUC") -> pd.DataFrame:
    """
    Fetches runs from the specified W&B project. Filters by group if provided.
    Finds the best validation score and its corresponding test score, 
    and returns them as a DataFrame.
    """
    api = wandb.Api()
    
    # Set up filters dynamically based on whether a group_name is provided
    filters = {}
    if group_name:
        filters["group"] = group_name

    runs = api.runs(
        f"{entity}/{project}",
        filters=filters
    )

    if group_name:
        print(f"Found {len(runs)} runs in the '{group_name}' group.\n")
    else:
        print(f"Found {len(runs)} runs in the project '{project}'.\n")

    results = []

    for run in runs:
        history = run.history(samples=100000)

        # Drop rows where the validation metric is missing
        clean_history = history.dropna(subset=[val_metric])
        if clean_history.empty:
            continue

        # Find the row with the best validation metric
        best_row = clean_history.loc[clean_history[val_metric].idxmax()]
        best_step = best_row["_step"]

        # Drop rows where the test metric is missing
        test_history = history.dropna(subset=[test_metric])
        
        if not test_history.empty:
            # Find the test step closest to the best validation step
            closest_idx = (test_history["_step"] - best_step).abs().idxmin()
            test_value = test_history.loc[closest_idx, test_metric]
            test_step = test_history.loc[closest_idx, "_step"]
        else:
            test_value = "N/A"
            test_step = "N/A"

        results.append({
            "run": run.name,
            f"best_val_{val_metric}": best_row[val_metric],
            f"test_{test_metric}": test_value,
            "val_step": best_step,
            "test_step": test_step
        })

    return pd.DataFrame(results)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Fetch best W&B runs based on validation metrics.")
    
    # Required arguments
    parser.add_argument("--entity", required=True, type=str, help="W&B Entity (username or team name)")
    parser.add_argument("--project", required=True, type=str, help="W&B Project name")
    
    # Optional arguments
    parser.add_argument("--group", type=str, default=None, help="W&B Group name (optional)")
    parser.add_argument("--val-metric", type=str, default="Validation/ROC AUC", help="Validation metric name to maximize")
    parser.add_argument("--test-metric", type=str, default="Test/ROC AUC", help="Test metric name to track")
    
    # Parse the arguments from the console
    args = parser.parse_args()
    
    print(f"Fetching data from W&B for project '{args.project}', please wait...\n")
    
    # Run the function with console arguments
    df = get_best_wandb_runs(
        entity=args.entity,
        project=args.project,
        group_name=args.group,
        val_metric=args.val_metric,
        test_metric=args.test_metric
    )
    
    # Print the results to the console
    if not df.empty:
        print("--- Best Results ---")
        print(df.to_string())
        
        # ---------------------------------------------------------
        # NEWLY ADDED SECTION: Statistical Calculations (Mean & Std)
        # ---------------------------------------------------------
        val_col_name = f"best_val_{args.val_metric}"
        test_col_name = f"test_{args.test_metric}"
        
        # Convert string values like 'N/A' to NaN and cast columns to numeric format
        df[val_col_name] = pd.to_numeric(df[val_col_name], errors='coerce')
        df[test_col_name] = pd.to_numeric(df[test_col_name], errors='coerce')
        
        # Extract the base model name (group) by removing the '_seed_X' part from 'run' names
        df['model_base_name'] = df['run'].apply(lambda x: x.split('_seed_')[0])
        
        # Grouping and calculating statistics
        summary = df.groupby('model_base_name').agg({
            val_col_name: ['mean', 'std'],
            test_col_name: ['mean', 'std']
        }).reset_index()
        
        # Making column names more readable
        summary.columns = [
            'Model', 
            'Val Mean', 'Val Std', 
            'Test Mean', 'Test Std'
        ]
        
        print("\n--- Aggregated Results (Mean and Std) ---")
        print(summary.to_string(index=False))
        # ---------------------------------------------------------
        
    else:
        print("No valid results found matching the specified criteria and metrics.")

if __name__ == "__main__":
    main()