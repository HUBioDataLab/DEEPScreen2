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

        clean_history = history.dropna(subset=[val_metric])
        if clean_history.empty:
            continue

        best_row = clean_history.loc[clean_history[val_metric].idxmax()]
        best_step = best_row["_step"]

        test_history = history.dropna(subset=[test_metric])
        
        if not test_history.empty:
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
        
        # You can optionally save the results to a CSV file dynamically:
        # csv_suffix = args.group if args.group else "all"
        # csv_filename = f"wandb_results_{csv_suffix}.csv"
        # df.to_csv(csv_filename, index=False)
        # print(f"\nResults have been saved to {csv_filename}.")
    else:
        print("No valid results found matching the specified criteria and metrics.")

if __name__ == "__main__":
    main()